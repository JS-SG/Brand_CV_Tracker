"""
Lightweight Interactive Annotation Helper for CHASE Court Logo.

Allows rapid bounding-box annotation directly via OpenCV GUI without requiring
external annotation software. Generates standard single-class YOLO format labels:
`0 x_center y_center width height` (normalized 0.0 to 1.0).

Key Controls:
- Left-click and drag: Draw bounding box
- 's': Save label and advance to next image
- 'n' or Space: Mark image as negative / no CHASE court logo (creates empty label file)
- 'c': Clear current bounding box
- 'q' or Esc: Quit annotation session
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from typing import List, Optional, Tuple
import cv2
from src.utils import setup_logger

logger = setup_logger("annotate_helper")


class BBoxAnnotator:
    def __init__(self, images_dir: Path, labels_dir: Path) -> None:
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.labels_dir.mkdir(parents=True, exist_ok=True)

        self.image_extensions = {".jpg", ".jpeg", ".png", ".bmp"}
        self.image_files = sorted(
            [p for p in images_dir.iterdir() if p.suffix.lower() in self.image_extensions]
        )

        self.current_idx = 0
        self.start_point: Optional[Tuple[int, int]] = None
        self.end_point: Optional[Tuple[int, int]] = None
        self.drawing = False
        self.boxes: List[Tuple[int, int, int, int]] = []

    def _mouse_callback(self, event, x, y, flags, param) -> None:
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.start_point = (x, y)
            self.end_point = (x, y)
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.drawing:
                self.end_point = (x, y)
        elif event == cv2.EVENT_LBUTTONUP:
            self.drawing = False
            self.end_point = (x, y)
            if self.start_point and self.end_point:
                x1 = min(self.start_point[0], self.end_point[0])
                y1 = min(self.start_point[1], self.end_point[1])
                x2 = max(self.start_point[0], self.end_point[0])
                y2 = max(self.start_point[1], self.end_point[1])
                if (x2 - x1) > 5 and (y2 - y1) > 5:
                    self.boxes = [(x1, y1, x2, y2)]  # Single target logo
            self.start_point = None
            self.end_point = None

    def run(self) -> None:
        if not self.image_files:
            logger.info(f"No images found in {self.images_dir}")
            return

        window_name = "CHASE Court Logo Annotator (Draw box, 's'=Save, 'n'=Skip/Negative, 'c'=Clear, 'q'=Quit)"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(window_name, self._mouse_callback)

        logger.info(f"Found {len(self.image_files)} images to annotate.")
        logger.info("Controls: Drag box | 's' = Save | 'n'/'Space' = Skip/No logo | 'c' = Clear | 'q' = Quit")

        while self.current_idx < len(self.image_files):
            img_path = self.image_files[self.current_idx]
            label_path = self.labels_dir / f"{img_path.stem}.txt"

            frame = cv2.imread(str(img_path))
            if frame is None:
                self.current_idx += 1
                continue

            h, w = frame.shape[:2]
            self.boxes = []

            # Load existing label if present
            if label_path.exists():
                with open(label_path, "r", encoding="utf-8") as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) == 5:
                            _, xc, yc, bw, bh = map(float, parts)
                            x1 = int((xc - bw / 2.0) * w)
                            y1 = int((yc - bh / 2.0) * h)
                            x2 = int((xc + bw / 2.0) * w)
                            y2 = int((yc + bh / 2.0) * h)
                            self.boxes.append((x1, y1, x2, y2))

            while True:
                display = frame.copy()

                # Draw existing boxes
                for bx1, by1, bx2, by2 in self.boxes:
                    cv2.rectangle(display, (bx1, by1), (bx2, by2), (0, 255, 0), 2)
                    cv2.putText(
                        display,
                        "CHASE_COURT_LOGO",
                        (bx1, max(15, by1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 0),
                        2,
                    )

                # Draw active dragging box
                if self.drawing and self.start_point and self.end_point:
                    cv2.rectangle(display, self.start_point, self.end_point, (0, 165, 255), 2)

                # Status banner
                status_text = f"[{self.current_idx + 1}/{len(self.image_files)}] {img_path.name} | Boxes: {len(self.boxes)}"
                cv2.putText(
                    display,
                    status_text,
                    (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 0),
                    2,
                )

                cv2.imshow(window_name, display)
                key = cv2.waitKey(20) & 0xFF

                if key in (ord("q"), 27):  # Quit
                    cv2.destroyAllWindows()
                    logger.info("Annotation session ended by user.")
                    return

                elif key == ord("c"):  # Clear
                    self.boxes = []

                elif key in (ord("n"), 32):  # Skip / Mark negative
                    with open(label_path, "w", encoding="utf-8") as f:
                        pass  # empty file marks negative image
                    logger.info(f"Marked {img_path.name} as negative (no court logo).")
                    self.current_idx += 1
                    break

                elif key == ord("s"):  # Save
                    with open(label_path, "w", encoding="utf-8") as f:
                        for bx1, by1, bx2, by2 in self.boxes:
                            xc = ((bx1 + bx2) / 2.0) / w
                            yc = ((by1 + by2) / 2.0) / h
                            bw = (bx2 - bx1) / w
                            bh = (by2 - by1) / h
                            f.write(f"0 {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}\n")
                    logger.info(f"Saved annotation for {img_path.name}: {len(self.boxes)} box(es).")
                    self.current_idx += 1
                    break

        cv2.destroyAllWindows()
        logger.info("Completed annotations for all available images.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Interactive YOLO Annotation Tool for CHASE Court Logo.")
    parser.add_argument(
        "--images",
        type=str,
        default="dataset/images/train",
        help="Directory containing images to annotate",
    )
    parser.add_argument(
        "--labels",
        type=str,
        default="dataset/labels/train",
        help="Directory where YOLO .txt labels will be saved",
    )

    args = parser.parse_args()
    annotator = BBoxAnnotator(Path(args.images), Path(args.labels))
    annotator.run()


if __name__ == "__main__":
    main()
