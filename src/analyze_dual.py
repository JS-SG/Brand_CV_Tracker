"""
Simultaneous YOLO tracking of:
  1) CHASE court mid-court logo (existing, untouched model)
  2) State Farm logo on the PHYSICAL RIGHT-SIDE basket

The two assets use separate YOLO weights and separate IntervalTracker instances.
They are processed in the same video pass and written to separate CSV files.
The debug video shows both detections at the same time.
"""
from __future__ import annotations

import argparse, json, time
from pathlib import Path
from typing import Optional, Sequence

import cv2
import pandas as pd
import torch
from tqdm import tqdm
from ultralytics import YOLO

from interval_manager import Detection, IntervalTracker
from scene_detector import SceneChangeDetector
from utils import format_time, get_system_info, get_video_metadata, setup_logger

logger = setup_logger('analyze_dual')


def infer(model: YOLO, frame, conf: float, device: str, timestamp: float, frame_idx: int):
    out=[]
    results=model(frame, conf=conf, iou=0.30, max_det=5, imgsz=1280, device=device, verbose=False)
    for result in results:
        if result.boxes is None: continue
        for box in result.boxes:
            cls_id=int(box.cls[0].item()); score=float(box.conf[0].item())
            if cls_id != 0 or score < conf: continue
            x1,y1,x2,y2=map(float,box.xyxy[0].tolist())
            out.append(Detection(x1,y1,x2,y2,score,timestamp,frame_idx))
    return out


def center(d: Detection):
    return ((d.x1+d.x2)/2.0, (d.y1+d.y2)/2.0)


def choose_statefarm(candidates: Sequence[Detection], frame_w: int, chase: Sequence[Detection],
                     previous: Optional[Detection], policy: str) -> list[Detection]:
    """Select only one State Farm candidate, with an explicit right-basket policy."""
    if not candidates: return []
    if len(candidates)==1 and policy == 'highest-conf': return [candidates[0]]

    chase_c = center(max(chase,key=lambda d:d.confidence)) if chase else None
    scored=[]
    for d in candidates:
        cx,cy=center(d)
        score=d.confidence
        if policy == 'right-of-chase' and chase_c is not None:
            # Physical right-basket prior: target is on the right side of mid-court.
            # If a camera shot reverses this screen-side relationship, the model's
            # right-basket-only training examples remain the primary signal.
            dx=(cx-chase_c[0])/max(frame_w,1)
            score += 0.20 if dx > 0 else -0.20
        elif policy == 'image-right':
            score += 0.20 if cx >= frame_w*0.50 else -0.20
        if previous is not None:
            px,py=center(previous)
            dist=((cx-px)**2+(cy-py)**2)**0.5/max(frame_w,frame_w*0.75)
            score -= 0.35*dist
        scored.append((score,d))
    scored.sort(key=lambda x:x[0],reverse=True)
    return [scored[0][1]]


def write_csv(path: Path, intervals):
    cols=['appearance_id','start_time','end_time','duration_seconds','x1','y1','x2','y2','width','height','average_confidence','detection_count']
    rows=[x.to_dict() for x in intervals]
    pd.DataFrame(rows,columns=cols).to_csv(path,index=False)


def analyze(video: str, chase_model_path: str, statefarm_model_path: str,
            fps: float, chase_conf: float, statefarm_conf: float,
            max_missed: int, output_dir: str, device: str, save_debug_video: bool,
            save_detection_frames: bool, statefarm_policy: str):
    video_path=Path(video).resolve(); out=Path(output_dir).resolve(); out.mkdir(parents=True,exist_ok=True)
    if not video_path.exists(): raise FileNotFoundError(video_path)
    if not Path(chase_model_path).exists(): raise FileNotFoundError(f'CHASE model missing: {chase_model_path}')
    if not Path(statefarm_model_path).exists():
        raise FileNotFoundError(f'State Farm model missing: {statefarm_model_path}. Annotate/train it first.')

    if device=='auto': device='cuda:0' if torch.cuda.is_available() else 'cpu'
    meta=get_video_metadata(video_path); src_fps=meta['fps']; total=meta['frame_count']; w=meta['width']; h=meta['height']
    step=max(1,int(round(src_fps/fps))); actual_fps=src_fps/step
    chase_model=YOLO(str(Path(chase_model_path).resolve()))
    sf_model=YOLO(str(Path(statefarm_model_path).resolve()))
    scene=SceneChangeDetector(hist_threshold=0.55,diff_threshold=0.35)
    chase_tracker=IntervalTracker(max_missed=max_missed)
    sf_tracker=IntervalTracker(max_missed=max_missed)

    writer=None
    if save_debug_video:
        writer=cv2.VideoWriter(str(out/'dual_tracking.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),actual_fps,(w,h))
        if not writer.isOpened(): raise RuntimeError('Could not create output/dual_tracking.mp4')
    det_dir=out/'detections_dual'
    if save_detection_frames: det_dir.mkdir(parents=True,exist_ok=True)

    cap=cv2.VideoCapture(str(video_path)); idx=0; processed=0; total_time=0.0
    previous_sf=None
    chase_appearance_no = 0
    sf_appearance_no = 0

    chase_was_visible = False
    sf_was_visible = False
    pbar=tqdm(total=total,desc='Dual YOLO analysis',unit='frame')
    while True:
        ok,frame=cap.read()
        if not ok: break
        if idx % step == 0:
            ts=idx/src_fps; t0=time.perf_counter()
            is_cut,cut_score=scene.is_scene_change(frame)
            if is_cut:
                chase_tracker.handle_scene_change(ts); sf_tracker.handle_scene_change(ts); previous_sf=None

            chase_dets=infer(chase_model,frame,chase_conf,device,ts,idx)
            sf_candidates=infer(sf_model,frame,statefarm_conf,device,ts,idx)
            sf_dets=choose_statefarm(sf_candidates,w,chase_dets,previous_sf,statefarm_policy)
            if sf_dets: previous_sf=sf_dets[0]
            chase_visible = bool(chase_dets)
            sf_visible = bool(sf_dets)

            if chase_visible and not chase_was_visible:
                chase_appearance_no += 1

            if sf_visible and not sf_was_visible:
                sf_appearance_no += 1

            chase_was_visible = chase_visible
            sf_was_visible = sf_visible

            chase_tracker.update(ts,idx,chase_dets)
            sf_tracker.update(ts,idx,sf_dets)
            total_time += time.perf_counter()-t0; processed += 1

            if writer or save_detection_frames:
                vis=frame.copy()
                for d in chase_dets:
                    x1,y1,x2,y2=map(int,d.bbox)
                    cv2.rectangle(vis,(x1,y1),(x2,y2),(255,120,0),3)
                    cv2.putText(vis,f'CHASE COURT | {chase_appearance_no}',(x1,max(25,y1-8)),cv2.FONT_HERSHEY_SIMPLEX,.65,(255,120,0),2)
                for d in sf_dets:
                    x1,y1,x2,y2=map(int,d.bbox)
                    cv2.rectangle(vis,(x1,y1),(x2,y2),(0,180,255),3)
                    cv2.putText(vis,f'STATEFARM | {sf_appearance_no}',(x1,max(25,y1-8)),cv2.FONT_HERSHEY_SIMPLEX,.65,(0,180,255),2)
                cv2.rectangle(vis,(10,10),(720,92),(0,0,0),-1)
                cv2.putText(vis,f'Time {format_time(ts)} | Frame {idx} | CHASE: {"VISIBLE" if chase_dets else "not visible"} | StateFarm: {"VISIBLE" if sf_dets else "not visible"}',(20,38),cv2.FONT_HERSHEY_SIMPLEX,.65,(255,255,255),2)
                if writer: writer.write(vis)
                if save_detection_frames and (chase_dets or sf_dets):
                    cv2.imwrite(str(det_dir/f'frame_{idx:06d}_t{ts:.2f}s.jpg'),vis)
        idx += 1; pbar.update(1)
    pbar.close(); cap.release()
    if writer: writer.release()

    chase=chase_tracker.finalize(); sf=sf_tracker.finalize()
    chase_csv=out/'chase_intervals.csv'; sf_csv=out/'statefarm_intervals.csv'
    write_csv(chase_csv,chase); write_csv(sf_csv,sf)
    summary={
        'video':str(video_path),'video_duration_seconds':meta['duration_seconds'],'fps':actual_fps,
        'device':device,'hardware':get_system_info(),'average_processing_time_per_sampled_frame_seconds':round(total_time/max(processed,1),5),
        'assets':{
            'chase':{'model':str(Path(chase_model_path).resolve()),'interval_csv':str(chase_csv),'total_intervals':len(chase),'visible_duration_seconds':round(sum(x.duration_seconds for x in chase),3)},
            'statefarm_right_basket':{'model':str(Path(statefarm_model_path).resolve()),'selection_policy':statefarm_policy,'interval_csv':str(sf_csv),'total_intervals':len(sf),'visible_duration_seconds':round(sum(x.duration_seconds for x in sf),3)}
        },
        'debug_video':str(out/'dual_tracking.mp4') if save_debug_video else None
    }
    (out/'dual_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print('\n'+'='*100)
    print('SIMULTANEOUS CHASE + STATE FARM RIGHT-BASKET TRACKING')
    print('='*100)
    print(f'Video: {video_path.name} | Duration: {meta["duration_seconds"]:.2f}s | Effective FPS: {actual_fps:.2f}')
    print(f'CHASE:     {len(chase)} intervals | {sum(x.duration_seconds for x in chase):.3f}s | CSV: {chase_csv}')
    print(f'STATE FARM: {len(sf)} intervals | {sum(x.duration_seconds for x in sf):.3f}s | CSV: {sf_csv}')
    print(f'Debug video: {out/"dual_tracking.mp4" if save_debug_video else "disabled"}')
    print('='*100)
    return summary

def main():
    p = argparse.ArgumentParser(
        description='Simultaneous YOLO tracking of CHASE court logo and State Farm right-basket logo.'
    )

    p.add_argument('--video', default='data/source_video.mp4')

    p.add_argument(
        '--chase_model_path',
        default='models/chase_court_logo.pt'
    )

    p.add_argument(
        '--statefarm_model_path',
        default='models/statefarm_right_basket.pt'
    )

    p.add_argument('--fps', type=float, default=10)
    p.add_argument('--chase_conf', type=float, default=.45)
    p.add_argument('--statefarm_conf', type=float, default=.35)
    p.add_argument('--max_missed', type=int, default=2)
    p.add_argument('--output_dir', default='output')
    p.add_argument('--device', default='auto')
    p.add_argument('--save_debug_video', action='store_true')
    p.add_argument('--save_detection_frames', action='store_true')

    p.add_argument(
        '--statefarm_policy',
        choices=['right-of-chase', 'image-right', 'highest-conf'],
        default='highest-conf'
    )

    a = p.parse_args()
    analyze(**vars(a))


if __name__ == '__main__':
    main()

