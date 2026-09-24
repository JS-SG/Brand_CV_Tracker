"""
Frame Extractor for Training Dataset Creation.

Extracts frames from the target sports broadcast video at a configurable sampling interval
so the developer can annotate frames containing the CHASE court logo.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
from tqdm import tqdm
from src.utils import get_video_metadata, setup_logger

logger = setup_logger("extract_training_frames")


def extract_frames(
    video_path: str | Path,
    output_dir: str | Path,
    interval_seconds: float = 1.0,
    max_frames: int | None = None,
    prefix: str = "court_frame",
) -> int:
    """
    Extracts frames from video at a regular time interval in seconds.

    Args:
        video_path: Path to source video file.
        output_dir: Directory where extracted images will be saved.
        interval_seconds: Time interval between extracted frames in seconds.
        max_frames: Optional upper limit on extracted frames.
        prefix: Filename prefix for saved images.

    Returns:
        Total number of frames extracted.
    """
    video_file = Path(video_path).resolve()
    out_dir = Path(output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    meta = get_video_metadata(video_file)
    fps = meta["fps"]
    frame_count = meta["frame_count"]
    duration = meta["duration_seconds"]

    step_frames = max(1, int(round(fps * interval_seconds)))

    logger.info(f"Source video: {video_file.name}")
    logger.info(f"Video metadata: {meta['width']}x{meta['height']} @ {fps:.2f} FPS ({duration:.2f}s total)")
    logger.info(f"Extracting 1 frame every {interval_seconds:.2f}s (~{step_frames} source frames)...")

    cap = cv2.VideoCapture(str(video_file))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_file}")

    extracted_count = 0
    curr_frame_idx = 0

    pbar = tqdm(total=frame_count, desc="Extracting frames", unit="frame")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if curr_frame_idx % step_frames == 0:
            timestamp_sec = curr_frame_idx / fps
            filename = f"{prefix}_{extracted_count:05d}_t{timestamp_sec:.2f}s.jpg"
            save_path = out_dir / filename
            cv2.imwrite(str(save_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
            extracted_count += 1

            if max_frames is not None and extracted_count >= max_frames:
                logger.info(f"Reached specified limit of {max_frames} frames.")
                break

        curr_frame_idx += 1
        pbar.update(1)

    pbar.close()
    cap.release()

    logger.info(f"Successfully extracted {extracted_count} frames to: {out_dir}")
    return extracted_count


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract training frames from basketball video at configurable intervals."
    )
    parser.add_argument(
        "--video",
        type=str,
        default="data/source_video.mp4",
        help="Path to source broadcast video",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="dataset/extracted_raw_frames",
        help="Directory to save extracted frame images",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=1.0,
        help="Sampling interval in seconds between extracted frames (default: 1.0s)",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Maximum number of frames to extract (optional)",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default="chase_sample",
        help="Prefix for saved image filenames",
    )

    args = parser.parse_args()

    try:
        extract_frames(
            video_path=args.video,
            output_dir=args.output_dir,
            interval_seconds=args.interval,
            max_frames=args.max_frames,
            prefix=args.prefix,
        )
    except Exception as exc:
        logger.error(f"Frame extraction failed: {exc}", exc_info=True)


if __name__ == "__main__":
    main()
