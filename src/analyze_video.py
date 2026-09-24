"""
Main Video Inference Pipeline for Static CHASE Court Logo Detection and Interval Tracking.

Processes broadcast video at configurable sampling FPS, applies YOLO detection,
identifies camera cuts via scene detector, and groups continuous appearances
into intervals with median bounding box calculations and tolerance for detector misses.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from typing import Any, Dict, List, Optional
import cv2
import pandas as pd
import torch
from tqdm import tqdm

from src.interval_manager import AppearanceInterval, Detection, IntervalTracker
from src.scene_detector import SceneChangeDetector
from src.utils import (
    format_time,
    get_system_info,
    get_video_metadata,
    render_summary_table,
    setup_logger,
)

logger = setup_logger("analyze_video")


def analyze_video(
    video_path: str | Path,
    model_path: str | Path = "models/chase_court_logo.pt",
    sampling_fps: float = 10.0,
    confidence_threshold: float = 0.45,
    max_missed: int = 2,
    output_dir: str | Path = "output",
    save_debug_video: bool = False,
    save_detection_frames: bool = False,
    device: str | None = None,
    allow_mock: bool = False,
) -> Dict[str, Any]:
    """
    Executes end-to-end analysis on the sports broadcast video.

    Args:
        video_path: Path to broadcast video file.
        model_path: Path to trained YOLO model weights.
        sampling_fps: Frames per second to sample for inference.
        confidence_threshold: Minimum detection confidence to accept.
        max_missed: Maximum tolerated consecutive missed sample frames before closing interval.
        output_dir: Output directory for CSV, JSON, and visual artifacts.
        save_debug_video: If True, writes output/debug_tracking.mp4.
        save_detection_frames: If True, saves positive detection frames in output/detections/.
        device: 'cuda', 'cpu', or None for auto-detection.
        allow_mock: If True and model_path does not exist, uses mock detector for testing.

    Returns:
        Dictionary containing summary metrics and appearance interval list.
    """
    video_file = Path(video_path).resolve()
    model_file = Path(model_path).resolve()
    out_dir = Path(output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if not video_file.exists():
        raise FileNotFoundError(f"Input video not found: {video_file}")

    # Inspect environment and hardware
    sys_info = get_system_info()
    if device is None or device == "auto":
        device = "cuda:0" if torch.cuda.is_available() else "cpu"

    hardware_str = f"{sys_info['cuda_device_name']} (PyTorch {sys_info['pytorch_version']}, CUDA: {sys_info['cuda_available']})"

    logger.info("=" * 70)
    logger.info("INITIALIZING CHASE COURT LOGO ANALYSIS")
    logger.info(f"Target Brand / Asset: CHASE / Court Floor Logo")
    logger.info(f"Input Video: {video_file.name}")
    logger.info(f"Hardware: {hardware_str}")
    logger.info(f"Device: {device}")
    logger.info(f"Sampling FPS: {sampling_fps} | Confidence Threshold: {confidence_threshold}")
    logger.info(f"Missed Sample Tolerance: {max_missed} frames")
    logger.info("=" * 70)

    # Model resolution
    yolo_model = None
    if not model_file.exists():
        if allow_mock:
            logger.warning(
                f"Model file '{model_file.name}' not found. Running in MOCK test mode (--allow-mock)."
            )
        else:
            error_msg = (
                f"\nTrained model weights not found at: {model_file}\n\n"
                f"To train the one-class CHASE court logo detector, run:\n"
                f"  1. Extract frames:  python src/extract_training_frames.py --video {video_file}\n"
                f"  2. Annotate logo:   python src/annotate_helper.py\n"
                f"  3. Train model:     python src/train.py --data dataset/data.yaml --output {model_file}\n\n"
                f"For integration pipeline testing without a trained model, pass: --allow-mock\n"
            )
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)
    else:
        from ultralytics import YOLO
        logger.info(f"Loading custom YOLO weights: {model_file}")
        yolo_model = YOLO(str(model_file))

    # Video properties
    meta = get_video_metadata(video_file)
    src_fps = meta["fps"]
    total_frames = meta["frame_count"]
    width = meta["width"]
    height = meta["height"]
    duration_sec = meta["duration_seconds"]

    logger.info(f"Video Stream: {width}x{height} @ {src_fps:.2f} FPS | Total Frames: {total_frames} | Duration: {duration_sec:.2f}s")

    # Frame step calculation
    step_frames = max(1, int(round(src_fps / sampling_fps)))
    actual_sampling_fps = src_fps / step_frames
    logger.info(f"Processing every {step_frames} frame(s) (~{actual_sampling_fps:.2f} effective FPS)")

    # Initialize modules
    scene_detector = SceneChangeDetector(hist_threshold=0.55, diff_threshold=0.35)
    interval_tracker = IntervalTracker(max_missed=max_missed)

    # Optional debug outputs
    debug_writer = None
    if save_debug_video:
        debug_video_path = out_dir / "debug_tracking.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        debug_writer = cv2.VideoWriter(
            str(debug_video_path), fourcc, actual_sampling_fps, (width, height)
        )
        logger.info(f"Recording visual tracking debug video to: {debug_video_path}")

    detections_dir = out_dir / "detections"
    if save_detection_frames:
        detections_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Saving detection snapshots to: {detections_dir}")

    # Processing loop
    cap = cv2.VideoCapture(str(video_file))
    processed_count = 0
    curr_frame_idx = 0

    pbar = tqdm(total=total_frames, desc="Analyzing video", unit="frame")
    total_proc_time = 0.0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if curr_frame_idx % step_frames == 0:
            timestamp = curr_frame_idx / src_fps
            t0 = time.perf_counter()

            # 1. Scene cut detection (fast histogram & frame difference)
            is_cut, cut_score = scene_detector.is_scene_change(frame)
            if is_cut:
                closed_inv = interval_tracker.handle_scene_change(timestamp)
                if closed_inv:
                    logger.debug(
                        f"Scene cut detected at {format_time(timestamp)} (score {cut_score:.2f}). "
                        f"Closed interval #{closed_inv.appearance_id}."
                    )

            # 2. YOLO Model Inference
            valid_detections: List[Detection] = []
            if len(valid_detections) > 1:
                valid_detections = [max(valid_detections, key=lambda d: d.confidence)]
            if yolo_model is not None:
                results = yolo_model(
                    frame,
                    conf=confidence_threshold,
                    iou=0.3,
                    max_det=3,
                    imgsz=1280,
                    device=device,
                    verbose=False,
                )
                for res in results:
                    boxes = res.boxes
                    for box in boxes:
                        cls_id = int(box.cls[0].item())
                        conf = float(box.conf[0].item())
                        # Single-class detector (chase_court_logo = 0)
                        if cls_id == 0 and conf >= confidence_threshold:
                            xyxy = box.xyxy[0].tolist()
                            valid_detections.append(
                                Detection(
                                    x1=xyxy[0],
                                    y1=xyxy[1],
                                    x2=xyxy[2],
                                    y2=xyxy[3],
                                    confidence=conf,
                                    timestamp=timestamp,
                                    frame_idx=curr_frame_idx,
                                )
                            )
            elif allow_mock:
                # Synthetic mock detection generator for pipeline validation
                # Emulates periodic court appearance
                cycle = int(timestamp) % 8
                if 2 <= cycle <= 5 and not is_cut:
                    valid_detections.append(
                        Detection(
                            x1=float(width * 0.35),
                            y1=float(height * 0.45),
                            x2=float(width * 0.50),
                            y2=float(height * 0.50),
                            confidence=0.88,
                            timestamp=timestamp,
                            frame_idx=curr_frame_idx,
                        )
                    )

            # 3. Update Temporal Interval Tracker
            interval_tracker.update(
                timestamp=timestamp,
                frame_idx=curr_frame_idx,
                detections=valid_detections,
            )

            # Benchmark timing
            t1 = time.perf_counter()
            total_proc_time += (t1 - t0)
            processed_count += 1

            # 4. Optional Visual Debug Annotations
            if save_debug_video or save_detection_frames:
                vis_frame = frame.copy()
                is_active = interval_tracker.is_active
                active_id = (
                    interval_tracker.next_appearance_id if is_active else None
                )

                if valid_detections:
                    for det in valid_detections:
                        bx1, by1, bx2, by2 = map(int, det.bbox)
                        cv2.rectangle(vis_frame, (bx1, by1), (bx2, by2), (255, 120, 0), 3)
                        label = f"CHASE COURT LOGO #{active_id} | Conf: {det.confidence:.2f}"
                        cv2.putText(
                            vis_frame,
                            label,
                            (bx1, max(25, by1 - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            (255, 120, 0),
                            2,
                        )

                # Info HUD overlay
                hud_text = f"Time: {format_time(timestamp)} | Frame: {curr_frame_idx} | Intervals: {len(interval_tracker.completed_intervals)}"
                cv2.putText(
                    vis_frame,
                    hud_text,
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 255),
                    2,
                )

                if debug_writer is not None:
                    debug_writer.write(vis_frame)

                if save_detection_frames and valid_detections:
                    snap_path = detections_dir / f"frame_{curr_frame_idx:06d}_t{timestamp:.2f}s.jpg"
                    cv2.imwrite(str(snap_path), vis_frame)

        curr_frame_idx += 1
        pbar.update(1)

    pbar.close()
    cap.release()

    if debug_writer is not None:
        debug_writer.release()
        logger.info(f"Saved tracking debug video.")

    # Finalize intervals at end of video
    completed_intervals = interval_tracker.finalize()

    # Calculate overall performance metrics
    avg_proc_time_per_frame = (
        total_proc_time / processed_count if processed_count > 0 else 0.0
    )
    total_visible_duration = sum(inv.duration_seconds for inv in completed_intervals)

    # Convert intervals to dictionary records
    interval_records = [inv.to_dict() for inv in completed_intervals]

    # Save CSV output
    csv_path = out_dir / "results.csv"
    if interval_records:
        df = pd.DataFrame(interval_records)
    else:
        df = pd.DataFrame(
            columns=[
                "appearance_id",
                "start_time",
                "end_time",
                "duration_seconds",
                "x1",
                "y1",
                "x2",
                "y2",
                "width",
                "height",
                "average_confidence",
                "detection_count",
            ]
        )
    df.to_csv(csv_path, index=False)
    logger.info(f"Saved results CSV: {csv_path}")

    # Save JSON output
    json_path = out_dir / "results.json"
    results_json_data = {
        "video": str(video_file),
        "target_brand": "CHASE",
        "target_asset": "CHASE court mid-court logo",
        "video_duration_seconds": duration_sec,
        "sampling_fps": sampling_fps,
        "total_intervals": len(completed_intervals),
        "total_visible_duration_seconds": round(total_visible_duration, 3),
        "average_processing_time_per_frame_seconds": round(avg_proc_time_per_frame, 5),
        "hardware": hardware_str,
        "model": str(model_file),
        "intervals": interval_records,
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results_json_data, f, indent=2)
    logger.info(f"Saved results JSON: {json_path}")

    # Print terminal summary table
    summary_table = render_summary_table(
        video_path=video_file.name,
        target_asset="CHASE court mid-court logo",
        duration_seconds=duration_sec,
        sampling_fps=actual_sampling_fps,
        intervals=interval_records,
        avg_proc_time=avg_proc_time_per_frame,
        hardware_str=hardware_str,
        model_name=model_file.name if yolo_model else "Mock Detector",
    )
    print("\n" + summary_table + "\n")

    return results_json_data


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze basketball broadcast video to track static CHASE court logo."
    )
    parser.add_argument(
        "--video",
        type=str,
        default="data/source_video.mp4",
        help="Path to input basketball broadcast video",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="models/chase_court_logo.pt",
        help="Path to trained YOLO model weights",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=10.0,
        help="Sampling FPS for frame processing (default: 10.0)",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.45,
        help="Confidence threshold for court logo detections (default: 0.45)",
    )
    parser.add_argument(
        "--max-missed",
        type=int,
        default=2,
        help="Maximum tolerated consecutive missed samples (default: 2)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output",
        help="Directory to save results.csv, results.json, and debug artifacts",
    )
    parser.add_argument(
        "--save-debug-video",
        action="store_true",
        help="Save tracking debug video with annotations (output/debug_tracking.mp4)",
    )
    parser.add_argument(
        "--save-detection-frames",
        action="store_true",
        help="Save positive detection snapshot frames into output/detections/",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Execution device ('cuda', 'cpu', or 'auto')",
    )
    parser.add_argument(
        "--allow-mock",
        action="store_true",
        help="Enable mock detection fallback for integration testing when .pt weights are absent",
    )

    args = parser.parse_args()

    try:
        analyze_video(
            video_path=args.video,
            model_path=args.model,
            sampling_fps=args.fps,
            confidence_threshold=args.confidence,
            max_missed=args.max_missed,
            output_dir=args.output_dir,
            save_debug_video=args.save_debug_video,
            save_detection_frames=args.save_detection_frames,
            device=args.device,
            allow_mock=args.allow_mock,
        )
    except Exception as exc:
        logger.error(f"Analysis failed: {exc}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
