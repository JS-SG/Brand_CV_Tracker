"""
Lightweight scene-change detector using color histogram and frame difference.

Avoids deep-learning overhead to guarantee sub-millisecond per-frame cut detection.
"""

from __future__ import annotations

from typing import Optional, Tuple
import cv2
import numpy as np


class SceneChangeDetector:
    """
    Detects camera cuts and abrupt scene transitions using HSV histogram correlation
    and normalized frame difference on downscaled frames.
    """

    def __init__(
        self,
        hist_threshold: float = 0.55,
        diff_threshold: float = 0.35,
        resize_dim: Tuple[int, int] = (160, 90),
    ) -> None:
        """
        Args:
            hist_threshold: Correlation below this value triggers a scene cut (lower = more different).
            diff_threshold: Mean absolute pixel difference above this triggers a scene cut.
            resize_dim: Downscaled resolution (width, height) for ultra-fast comparison.
        """
        self.hist_threshold = hist_threshold
        self.diff_threshold = diff_threshold
        self.resize_dim = resize_dim

        self._prev_frame_gray: Optional[np.ndarray] = None
        self._prev_hist: Optional[np.ndarray] = None

    def reset(self) -> None:
        """Resets the historical frame state."""
        self._prev_frame_gray = None
        self._prev_hist = None

    def is_scene_change(self, frame: np.ndarray) -> Tuple[bool, float]:
        """
        Evaluates whether the current frame is a camera cut compared to the previous frame.

        Args:
            frame: BGR numpy image frame.

        Returns:
            Tuple of (is_cut: bool, difference_metric: float)
            Where difference_metric is a normalized score between 0.0 (identical) and 1.0 (completely different).
        """
        if frame is None or frame.size == 0:
            return False, 0.0

        # Downscale for performance (< 0.5 ms)
        small_frame = cv2.resize(frame, self.resize_dim, interpolation=cv2.INTER_LINEAR)
        gray = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)

        # Compute HSV histogram (H: 30 bins, S: 32 bins)
        hsv = cv2.cvtColor(small_frame, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None, [30, 32], [0, 180, 0, 256])
        cv2.normalize(hist, hist, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)

        if self._prev_hist is None or self._prev_frame_gray is None:
            self._prev_hist = hist
            self._prev_frame_gray = gray
            return False, 0.0

        # 1. Histogram correlation comparison: 1.0 = identical, -1.0 or 0.0 = completely different
        hist_corr = float(cv2.compareHist(self._prev_hist, hist, cv2.HISTCMP_CORREL))

        # 2. Normalized Mean Absolute Difference on grayscale
        diff = cv2.absdiff(self._prev_frame_gray, gray)
        norm_diff = float(np.mean(diff) / 255.0)

        # Composite difference score: 0.0 (same) to 1.0 (cut)
        # Invert hist_corr into [0, 1] distance: (1 - max(0, hist_corr))
        hist_dist = max(0.0, 1.0 - max(0.0, hist_corr))
        composite_score = 0.6 * hist_dist + 0.4 * norm_diff

        is_cut = (hist_corr < self.hist_threshold) or (norm_diff > self.diff_threshold)

        # Update cache for next iteration
        self._prev_hist = hist
        self._prev_frame_gray = gray

        return is_cut, composite_score
