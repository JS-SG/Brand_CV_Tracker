"""
Extract exactly N candidate frames for State Farm right-basket annotation.

This script does not assume fixed timestamps. It samples the source video at a
regular interval and saves candidate frames. Review the candidates and keep
frames where the RIGHT-SIDE PHYSICAL basket State Farm logo is visible.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import cv2


def extract_candidates(video: str, output_dir: str, count: int = 3, interval: float = 15.0, start: float = 0.0) -> int:
    video_path = Path(video)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS)) or 30.0
    duration = (float(cap.get(cv2.CAP_PROP_FRAME_COUNT)) / fps)
    saved = 0
    t = max(0.0, start)
    while saved < count and t <= duration:
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000.0)
        ok, frame = cap.read()
        if not ok:
            t += interval
            continue
        frame_idx = int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1
        path = out / f"statefarm_candidate_{saved:02d}_t{t:.2f}s_f{frame_idx}.jpg"
        cv2.imwrite(str(path), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
        print(f"saved: {path}")
        saved += 1
        t += interval
    cap.release()
    print(f"Extracted {saved} candidate frames from {video_path}")
    return saved


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--video", default="data/source_video.mp4")
    p.add_argument("--output-dir", default="dataset_statefarm/raw")
    p.add_argument("--count", type=int, default=3)
    p.add_argument("--interval", type=float, default=15.0)
    p.add_argument("--start", type=float, default=0.0)
    args = p.parse_args()
    extract_candidates(args.video, args.output_dir, args.count, args.interval, args.start)


if __name__ == "__main__":
    main()
