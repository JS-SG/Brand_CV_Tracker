"""
Temporal Interval Manager for CHASE Court Logo Appearance Tracking.

Distinguishes between:
1. Detector uncertainty: 1 to max_missed consecutive missed frames -> Keep interval active
2. Genuine disappearance: > max_missed consecutive missed frames due to occlusion,
   zoom, framing changes, or moving out-of-frame -> Finalize and close interval
3. Camera cut: Strong scene change detected -> Immediately finalize and close interval
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple
import numpy as np


@dataclass
class Detection:
    """Represents a single detected bounding box in a sampled frame."""
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float
    timestamp: float
    frame_idx: int

    @property
    def bbox(self) -> Tuple[float, float, float, float]:
        return (self.x1, self.y1, self.x2, self.y2)


@dataclass
class AppearanceInterval:
    """
    Represents a verified continuous appearance interval of the physical
    static CHASE logo printed on the court floor.
    """
    appearance_id: int
    start_time: float
    end_time: float
    duration_seconds: float
    x1: float
    y1: float
    x2: float
    y2: float
    width: float
    height: float
    average_confidence: float
    detection_count: int
    detections: List[Detection] = field(default_factory=list, repr=False)

    def to_dict(self) -> Dict[str, Any]:
        """Converts interval to standard dictionary representation."""
        return {
            "appearance_id": self.appearance_id,
            "start_time": round(self.start_time, 3),
            "end_time": round(self.end_time, 3),
            "duration_seconds": round(self.duration_seconds, 3),
            "x1": round(self.x1, 2),
            "y1": round(self.y1, 2),
            "x2": round(self.x2, 2),
            "y2": round(self.y2, 2),
            "width": round(self.width, 2),
            "height": round(self.height, 2),
            "average_confidence": round(self.average_confidence, 4),
            "detection_count": self.detection_count,
        }


class IntervalTracker:
    """
    Tracks and aggregates logo detections into appearance intervals across time.

    Key Logic:
    - Maintains state: active interval, start_time, last_detection_time, detections, missed_count.
    - If logo detected:
        * Starts new interval if none active.
        * Continues active interval if already running.
        * Resets missed_count to 0.
    - If logo missed:
        * Increments missed_count.
        * If missed_count <= max_missed: Treats as temporary detector uncertainty and retains interval.
        * If missed_count > max_missed: Treats as genuine disappearance (occlusion, zoom, framing change)
          and closes the active interval with end_time = last_detection_time.
    - If camera cut:
        * Closes active interval immediately at last_detection_time.
        * Resets tracking state so subsequent appearances start fresh intervals.
    """

    def __init__(self, max_missed: int = 2) -> None:
        """
        Args:
            max_missed: Maximum tolerated consecutive missed sample frames before
                        closing the active interval.
        """
        self.max_missed = max_missed

        # State tracking
        self.active_start_time: Optional[float] = None
        self.active_last_time: Optional[float] = None
        self.active_detections: List[Detection] = []
        self.missed_count: int = 0
        self.next_appearance_id: int = 1

        self.completed_intervals: List[AppearanceInterval] = []

    @property
    def is_active(self) -> bool:
        """Returns True if there is currently an open appearance interval."""
        return self.active_start_time is not None

    def update(
        self,
        timestamp: float,
        frame_idx: int,
        detections: Sequence[Detection],
    ) -> Optional[AppearanceInterval]:
        """
        Processes detections for the current sampled frame.

        Args:
            timestamp: Video timestamp in seconds for this sampled frame.
            frame_idx: Integer frame index.
            detections: List of valid CHASE court logo detections in this frame.

        Returns:
            An AppearanceInterval instance if an interval was finalized and closed
            during this update step, otherwise None.
        """
        closed_interval: Optional[AppearanceInterval] = None

        if len(detections) > 0:
            # Pick highest confidence detection if multiple are detected
            best_detection = max(detections, key=lambda d: d.confidence)

            if not self.is_active:
                # Start new appearance interval
                self.active_start_time = timestamp
                self.active_last_time = timestamp
                self.active_detections = [best_detection]
                self.missed_count = 0
            else:
                # Continue active appearance interval
                self.active_last_time = timestamp
                self.active_detections.append(best_detection)
                self.missed_count = 0
        else:
            # No detection in this frame
            if self.is_active:
                self.missed_count += 1
                if self.missed_count > self.max_missed:
                    # Genuine disappearance: logo no longer visible
                    # (due to occlusion, zoom, framing, or moving out of frame)
                    closed_interval = self._close_active_interval()

        return closed_interval

    def handle_scene_change(self, timestamp: float) -> Optional[AppearanceInterval]:
        """
        Handles camera cuts / scene changes by immediately closing any open interval.

        Args:
            timestamp: Timestamp in seconds where the scene cut was detected.

        Returns:
            The finalized AppearanceInterval if an interval was active, else None.
        """
        if self.is_active:
            closed_interval = self._close_active_interval()
            return closed_interval
        return None

    def finalize(self) -> List[AppearanceInterval]:
        """
        Finalizes tracking at the end of video processing. Closes any lingering active interval.

        Returns:
            Complete list of all completed AppearanceInterval instances.
        """
        if self.is_active:
            self._close_active_interval()
        return list(self.completed_intervals)

    def _close_active_interval(self) -> AppearanceInterval:
        """Internal helper to calculate summary stats and close the active interval."""
        if not self.is_active or self.active_start_time is None or self.active_last_time is None:
            raise RuntimeError("Cannot close interval when no interval is active.")

        start_t = self.active_start_time
        end_t = self.active_last_time
        duration = round(max(0.0, end_t - start_t), 3)

        # Median bounding box calculation
        raw_bboxes = [d.bbox for d in self.active_detections]
        arr = np.array(raw_bboxes, dtype=np.float64)
        med = np.median(arr, axis=0)
        x1, y1, x2, y2 = float(med[0]), float(med[1]), float(med[2]), float(med[3])
        width = max(0.0, x2 - x1)
        height = max(0.0, y2 - y1)

        avg_conf = float(np.mean([d.confidence for d in self.active_detections]))

        interval = AppearanceInterval(
            appearance_id=self.next_appearance_id,
            start_time=start_t,
            end_time=end_t,
            duration_seconds=duration,
            x1=round(x1, 2),
            y1=round(y1, 2),
            x2=round(x2, 2),
            y2=round(y2, 2),
            width=round(width, 2),
            height=round(height, 2),
            average_confidence=round(avg_conf, 4),
            detection_count=len(self.active_detections),
            detections=list(self.active_detections),
        )

        self.completed_intervals.append(interval)
        self.next_appearance_id += 1

        # Reset active tracking state
        self.active_start_time = None
        self.active_last_time = None
        self.active_detections = []
        self.missed_count = 0

        return interval
