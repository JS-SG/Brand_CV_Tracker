"""
YOLO Training Script for Single-Class CHASE Court Logo Detection.

Fine-tunes a lightweight nano model (YOLO11n / YOLOv8n) on the annotated CHASE court logo dataset.
Saves the best weights to models/chase_court_logo.pt.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
from src.utils import get_system_info, setup_logger

logger = setup_logger("train")


def train_chase_detector(
    data_yaml: str | Path = "dataset/data.yaml",
    base_model: str = "yolo11n.pt",
    epochs: int = 50,
    batch_size: int = 16,
    img_size: int = 640,
    device: str | None = None,
    output_model_path: str | Path = "models/chase_court_logo.pt",
    workers: int = 2,
) -> Path:
    """
    Trains YOLO nano on the single-class chase_court_logo dataset.

    Args:
        data_yaml: Path to data.yaml dataset descriptor.
        base_model: Starting lightweight model checkpoint (e.g. yolo11n.pt or yolov8n.pt).
        epochs: Number of training epochs.
        batch_size: Training batch size (default 16, ideal for RTX 4070 / RTX 3050).
        img_size: Input image resolution (default 640).
        device: 'cuda', 'cuda:0', 'cpu', or None for auto-detection.
        output_model_path: Destination path for best model weights.
        workers: DataLoader worker threads (set to 0 or 2 on Windows).

    Returns:
        Path to the saved best model weights.
    """
    data_path = Path(data_yaml).resolve()
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset config file not found: {data_path}")

    # Determine hardware acceleration
    if device is None or device == "auto":
        device = "cuda:0" if torch.cuda.is_available() else "cpu"

    sys_info = get_system_info()
    logger.info("=" * 60)
    logger.info("CHASE COURT LOGO DETECTOR - MODEL TRAINING")
    logger.info("=" * 60)
    logger.info(f"System: {sys_info['os_platform']} | Python: {sys_info['python_version']}")
    logger.info(f"PyTorch: {sys_info['pytorch_version']} | CUDA Available: {sys_info['cuda_available']}")
    logger.info(f"Active Device: {device} ({sys_info['cuda_device_name']})")
    logger.info(f"Dataset Config: {data_path}")
    logger.info(f"Base Architecture: {base_model}")
    logger.info(f"Epochs: {epochs} | Batch Size: {batch_size} | Image Size: {img_size}")
    logger.info("=" * 60)

    try:
        from ultralytics import YOLO
    except ImportError:
        logger.error("Ultralytics is not installed. Please run: pip install ultralytics")
        sys.exit(1)

    # Initialize model
    try:
        model = YOLO(base_model)
    except Exception as e:
        fallback_model = "yolov8n.pt"
        logger.warning(f"Could not load {base_model} ({e}). Falling back to {fallback_model}...")
        model = YOLO(fallback_model)

    # Execute training
    results = model.train(
        data=str(data_path),
        epochs=epochs,
        batch=batch_size,
        imgsz=img_size,
        device=device,
        workers=workers,
        project="runs/detect",
        name="chase_court_logo",
        exist_ok=True,
        verbose=True,
        plots=True,
        single_cls=True,  # enforces single-class focus
    )

    # Locate best weights dynamically from training run
    save_dir = None
    if hasattr(model, "trainer") and model.trainer is not None:
        save_dir = Path(model.trainer.save_dir)
    elif hasattr(results, "save_dir"):
        save_dir = Path(results.save_dir)

    best_weights = None
    if save_dir:
        cand_best = save_dir / "weights" / "best.pt"
        cand_last = save_dir / "weights" / "last.pt"
        if cand_best.exists():
            best_weights = cand_best
        elif cand_last.exists():
            best_weights = cand_last

    # Fallback search if save_dir was not directly available
    if best_weights is None:
        matches = list(Path("runs").glob("**/weights/best.pt"))
        if matches:
            # Pick most recently modified best.pt
            best_weights = max(matches, key=lambda p: p.stat().st_mtime)

    if best_weights is None or not best_weights.exists():
        raise RuntimeError("Training finished but no model weights were generated.")

    dest_path = Path(output_model_path).resolve()
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best_weights, dest_path)
    logger.info(f"Saved trained best model to: {dest_path}")

    # Log metrics
    try:
        metrics = model.val()
        logger.info("=" * 60)
        logger.info("TRAINING VALIDATION METRICS:")
        logger.info(f"  Precision: {metrics.box.mp:.4f}")
        logger.info(f"  Recall:    {metrics.box.mr:.4f}")
        logger.info(f"  mAP@50:    {metrics.box.map50:.4f}")
        logger.info(f"  mAP@50-95: {metrics.box.map:.4f}")
        logger.info("=" * 60)
    except Exception as exc:
        logger.warning(f"Could not evaluate final validation metrics: {exc}")

    return dest_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Train YOLO detector for CHASE court logo.")
    parser.add_argument(
        "--data",
        type=str,
        default="dataset/data.yaml",
        help="Path to data.yaml dataset config",
    )
    parser.add_argument(
        "--base-model",
        type=str,
        default="yolo11n.pt",
        help="Base model architecture (default: yolo11n.pt)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Number of training epochs (default: 50)",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=16,
        help="Batch size (default: 16)",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Image size in pixels (default: 640)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to train on ('cuda', 'cpu', or 'auto')",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="models/chase_court_logo.pt",
        help="Path to save best trained model weights",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=2,
        help="DataLoader worker count (default: 2)",
    )

    args = parser.parse_args()

    try:
        train_chase_detector(
            data_yaml=args.data,
            base_model=args.base_model,
            epochs=args.epochs,
            batch_size=args.batch,
            img_size=args.imgsz,
            device=args.device,
            output_model_path=args.output,
            workers=args.workers,
        )
    except Exception as exc:
        logger.error(f"Training failed: {exc}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
