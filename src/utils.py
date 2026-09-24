"""
Utility functions for environment detection, video metadata extraction,
bounding box calculations, logging, and summary table formatting.
"""

from __future__ import annotations

import logging
import platform
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np


def setup_logger(name: str = "chase_tracker", level: int = logging.INFO) -> logging.Logger:
    """Configures and returns a standardized logger."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            fmt="[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger


def get_system_info() -> Dict[str, Any]:
    """
    Collects runtime environment information including Python, PyTorch,
    CUDA, GPU hardware, OpenCV, and Ultralytics versions.
    """
    info: Dict[str, Any] = {
        "python_version": platform.python_version(),
        "os_platform": platform.platform(),
        "pytorch_version": "N/A",
        "cuda_available": False,
        "cuda_device_name": "None",
        "opencv_version": "N/A",
        "ultralytics_version": "N/A",
    }

    try:
        import torch

        info["pytorch_version"] = torch.__version__
        info["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            info["cuda_device_name"] = torch.cuda.get_device_name(0)
        else:
            info["cuda_device_name"] = "CPU fallback"
    except ImportError:
        pass

    try:
        import cv2

        info["opencv_version"] = cv2.__version__
    except ImportError:
        pass

    try:
        import ultralytics

        info["ultralytics_version"] = ultralytics.__version__
    except ImportError:
        pass

    return info


def get_video_metadata(video_path: str | Path) -> Dict[str, Any]:
    """
    Reads video properties using OpenCV and calculates actual duration.

    Args:
        video_path: Path to the video file.

    Returns:
        Dict with keys: fps, frame_count, width, height, duration_seconds.
    """
    import cv2

    path_obj = Path(video_path)
    if not path_obj.exists():
        raise FileNotFoundError(f"Video file not found: {path_obj.resolve()}")

    cap = cv2.VideoCapture(str(path_obj))
    if not cap.isOpened():
        raise ValueError(f"OpenCV could not open video file: {path_obj.resolve()}")

    fps = float(cap.get(cv2.CAP_PROP_FPS))
    if fps <= 0.0:
        fps = 30.0  # safe fallback if metadata is corrupted
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration_seconds = round(frame_count / fps, 3) if fps > 0 else 0.0

    cap.release()

    return {
        "fps": fps,
        "frame_count": frame_count,
        "width": width,
        "height": height,
        "duration_seconds": duration_seconds,
    }


def calculate_median_bbox(
    bboxes: Sequence[Tuple[float, float, float, float] | List[float]]
) -> Tuple[float, float, float, float, float, float]:
    """
    Calculates the representative median bounding box from a sequence of bboxes.

    Args:
        bboxes: List of (x1, y1, x2, y2) bounding box coordinates.

    Returns:
        Tuple of (x1, y1, x2, y2, width, height) rounded to 2 decimal places.
    """
    if not bboxes:
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

    arr = np.array(bboxes, dtype=np.float64)
    med = np.median(arr, axis=0)
    x1, y1, x2, y2 = float(med[0]), float(med[1]), float(med[2]), float(med[3])
    width = max(0.0, x2 - x1)
    height = max(0.0, y2 - y1)

    return (
        round(x1, 2),
        round(y1, 2),
        round(x2, 2),
        round(y2, 2),
        round(width, 2),
        round(height, 2),
    )


def format_time(seconds: float) -> str:
    """Formats seconds into HH:MM:SS.mmm format."""
    total_ms = int(round(seconds * 1000))
    ms = total_ms % 1000
    total_seconds = total_ms // 1000
    s = total_seconds % 60
    total_minutes = total_seconds // 60
    m = total_minutes % 60
    h = total_minutes // 60
    return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"


def render_summary_table(
    video_path: str,
    target_asset: str,
    duration_seconds: float,
    sampling_fps: float,
    intervals: List[Dict[str, Any]],
    avg_proc_time: float,
    hardware_str: str,
    model_name: str,
) -> str:
    """
    Renders a clean, structured Unicode/ASCII terminal summary table
    incorporating both execution overview and detailed appearance interval records.
    """
    lines: List[str] = []
    width = 96
    lines.append("=" * width)
    lines.append(f"{'CHASE COURT LOGO DETECTION & TRACKING SUMMARY':^{width}}")
    lines.append("=" * width)

    # Overview table
    lines.append(f"{'PROPERTY':<32} | {'VALUE'}")
    lines.append("-" * 32 + "-+-" + "-" * (width - 35))
    lines.append(f"{'Target Brand / Asset':<32} | CHASE / {target_asset}")
    lines.append(f"{'Video Source':<32} | {video_path}")
    lines.append(f"{'Video Duration':<32} | {duration_seconds:.2f}s ({format_time(duration_seconds)})")
    lines.append(f"{'Sampling FPS':<32} | {sampling_fps:.2f} FPS")
    lines.append(f"{'Detection Model':<32} | {model_name}")
    lines.append(f"{'Hardware / Environment':<32} | {hardware_str}")
    lines.append(
        f"{'Avg Processing Time / Frame':<32} | {avg_proc_time:.4f}s "
        f"({(1.0 / avg_proc_time if avg_proc_time > 0 else 0):.1f} effective FPS)"
    )

    total_intervals = len(intervals)
    total_visible_duration = sum(item.get("duration_seconds", 0.0) for item in intervals)
    lines.append(f"{'Total Appearance Intervals':<32} | {total_intervals}")
    lines.append(
        f"{'Total Accumulated Visible Duration':<32} | {total_visible_duration:.2f}s "
        f"({(total_visible_duration / duration_seconds * 100 if duration_seconds > 0 else 0):.1f}% of broadcast)"
    )
    lines.append("=" * width)

    # Intervals detail table
    if intervals:
        lines.append(f"{'APPEARANCE INTERVALS BREAKDOWN':^{width}}")
        lines.append("-" * width)
        header = (
            f"{'ID':>4} | {'Start':>12} | {'End':>12} | {'Duration':>10} | "
            f"{'BBox [x1,y1,x2,y2]':>26} | {'W x H':>12} | {'Avg Conf':>8} | {'Dets':>5}"
        )
        lines.append(header)
        lines.append("-" * width)

        for inv in intervals:
            aid = inv.get("appearance_id", 0)
            st = format_time(inv.get("start_time", 0.0))
            et = format_time(inv.get("end_time", 0.0))
            dur = f"{inv.get('duration_seconds', 0.0):.2f}s"
            bbox_str = f"[{inv.get('x1', 0):.0f},{inv.get('y1', 0):.0f},{inv.get('x2', 0):.0f},{inv.get('y2', 0):.0f}]"
            w_h = f"{inv.get('width', 0):.0f}x{inv.get('height', 0):.0f}"
            conf = f"{inv.get('average_confidence', 0.0):.2f}"
            cnt = inv.get("detection_count", 0)

            row = (
                f"{aid:>4} | {st:>12} | {et:>12} | {dur:>10} | "
                f"{bbox_str:>26} | {w_h:>12} | {conf:>8} | {cnt:>5}"
            )
            lines.append(row)
        lines.append("-" * width)
    else:
        lines.append(f"{'No CHASE court logo appearances detected':^{width}}")
        lines.append("-" * width)

    lines.append("=" * width)
    return "\n".join(lines)
