"""
timeline_store.py
=================
Structured Temporal Evidence Store & Activity Timeline.
Implements the central data structure specified in Section 10 of the Core Research Conclusion:
  - Contiguous interval merging with confidence aggregation
  - Sensor modality & channel provenance tracking (Acc X/Y/Z, Gyro X/Y/Z)
  - Deterministic timeline indexing and interval querying
"""

import json
from typing import Dict, List, Any, Optional, Tuple
import numpy as np

from src.dataset_loader import TARGET_ACTIVITIES, ACTIVITY_DISPLAY_NAMES, ACTIVITY_MODALITIES, DEFAULT_CHANNELS


class ActivityInterval:
    def __init__(
        self,
        activity: str,
        start_time: float,
        end_time: float,
        confidence: float,
        modality: Optional[List[str]] = None,
        channels: Optional[List[str]] = None,
        raw_indices: Optional[List[int]] = None
    ):
        self.activity = activity
        self.start = float(start_time)
        self.end = float(end_time)
        self.duration = max(0.0, self.end - self.start)
        self.confidence = float(confidence)
        self.modality = modality or ACTIVITY_MODALITIES.get(activity, ["accelerometer", "gyroscope"])
        self.channels = channels or DEFAULT_CHANNELS
        self.raw_indices = raw_indices or []

    def to_dict(self) -> Dict[str, Any]:
        return {
            "activity": self.activity,
            "activity_display": ACTIVITY_DISPLAY_NAMES.get(self.activity, self.activity),
            "start": round(self.start, 1),
            "end": round(self.end, 1),
            "duration": round(self.duration, 1),
            "confidence": round(self.confidence, 4),
            "modality": self.modality,
            "channels": self.channels
        }


class ActivityTimeline:
    """
    Central Temporal Evidence Store indexing detected activity intervals for a sensor recording session.
    """
    def __init__(self, session_id: str = "session_001"):
        self.session_id = session_id
        self.intervals: List[ActivityInterval] = []
        self.session_start: float = 0.0
        self.session_end: float = 0.0
        self.total_duration: float = 0.0

    @classmethod
    def from_predictions(
        cls,
        timestamps: np.ndarray,
        predictions: np.ndarray,
        confidences: np.ndarray,
        session_id: str = "session_001",
        window_sec: float = 60.0,
        min_confidence_threshold: float = 0.35,
        merge_gap_sec: float = 60.0,
        time_origin: Optional[float] = None,
        session_duration: Optional[float] = None,
    ) -> 'ActivityTimeline':
        """
        Constructs an ActivityTimeline from frame-level model predictions,
        merging contiguous windows into coherent temporal intervals.
        """
        timeline = cls(session_id=session_id)
        if len(timestamps) == 0 or len(predictions) == 0:
            return timeline

        # Sort by timestamp
        sort_idx = np.argsort(timestamps)
        ts_sorted = timestamps[sort_idx]
        preds_sorted = predictions[sort_idx]
        confs_sorted = confidences[sort_idx]

        # Convert to relative seconds from start
        base_t0 = float(time_origin) if time_origin is not None else ts_sorted[0]
        rel_ts = ts_sorted - base_t0

        timeline.session_start = 0.0
        timeline.session_end = (
            max(0.0, float(session_duration))
            if session_duration is not None
            else float(rel_ts[-1] + window_sec)
        )
        timeline.total_duration = timeline.session_end - timeline.session_start

        # Group into contiguous intervals
        current_activity_idx = -1
        current_start = 0.0
        current_end = 0.0
        current_confs = []
        current_indices = []

        raw_intervals: List[ActivityInterval] = []

        for i, (t, pred, conf) in enumerate(zip(rel_ts, preds_sorted, confs_sorted)):
            if conf < min_confidence_threshold:
                continue

            t_start = float(t)
            t_end = float(t + window_sec)

            if current_activity_idx == -1:
                current_activity_idx = pred
                current_start = t_start
                current_end = t_end
                current_confs = [conf]
                current_indices = [i]
            elif pred == current_activity_idx and (t_start - current_end) <= merge_gap_sec:
                # Extend current interval
                current_end = max(current_end, t_end)
                current_confs.append(conf)
                current_indices.append(i)
            else:
                # Finalize previous interval
                act_name = TARGET_ACTIVITIES[current_activity_idx] if current_activity_idx < len(TARGET_ACTIVITIES) else "unknown"
                raw_intervals.append(
                    ActivityInterval(
                        activity=act_name,
                        start_time=current_start,
                        end_time=current_end,
                        confidence=float(np.mean(current_confs)),
                        raw_indices=current_indices
                    )
                )
                current_activity_idx = pred
                current_start = t_start
                current_end = t_end
                current_confs = [conf]
                current_indices = [i]

        if current_activity_idx != -1 and current_confs:
            act_name = TARGET_ACTIVITIES[current_activity_idx] if current_activity_idx < len(TARGET_ACTIVITIES) else "unknown"
            raw_intervals.append(
                ActivityInterval(
                    activity=act_name,
                    start_time=current_start,
                    end_time=current_end,
                    confidence=float(np.mean(current_confs)),
                    raw_indices=current_indices
                )
            )

        timeline.intervals = raw_intervals
        return timeline

    def get_intervals_for_activity(
        self,
        activity: str,
        t_min: Optional[float] = None,
        t_max: Optional[float] = None,
        min_confidence: float = 0.0
    ) -> List[ActivityInterval]:
        """Retrieves intervals matching target activity within optional temporal bounds."""
        res = []
        for interval in self.intervals:
            if interval.activity != activity:
                continue
            if interval.confidence < min_confidence:
                continue

            # Temporal overlap check
            i_start = interval.start
            i_end = interval.end

            if t_min is not None and i_end < t_min:
                continue
            if t_max is not None and i_start > t_max:
                continue

            # Clamp interval to bounds if specified
            clipped_start = max(i_start, t_min) if t_min is not None else i_start
            clipped_end = min(i_end, t_max) if t_max is not None else i_end

            if clipped_end > clipped_start:
                res.append(
                    ActivityInterval(
                        activity=interval.activity,
                        start_time=clipped_start,
                        end_time=clipped_end,
                        confidence=interval.confidence,
                        modality=interval.modality,
                        channels=interval.channels,
                        raw_indices=interval.raw_indices
                    )
                )
        return res

    def get_all_intervals(
        self,
        t_min: Optional[float] = None,
        t_max: Optional[float] = None
    ) -> List[ActivityInterval]:
        """Returns all intervals optionally filtered by time bounds."""
        if t_min is None and t_max is None:
            return self.intervals
        
        filtered = []
        for inv in self.intervals:
            if t_min is not None and inv.end < t_min:
                continue
            if t_max is not None and inv.start > t_max:
                continue
            c_start = max(inv.start, t_min) if t_min is not None else inv.start
            c_end = min(inv.end, t_max) if t_max is not None else inv.end
            if c_end > c_start:
                filtered.append(
                    ActivityInterval(
                        activity=inv.activity,
                        start_time=c_start,
                        end_time=c_end,
                        confidence=inv.confidence,
                        modality=inv.modality,
                        channels=inv.channels
                    )
                )
        return filtered

    def get_transitions(self) -> List[Dict[str, Any]]:
        """Identifies transitions between consecutive activity intervals."""
        transitions = []
        sorted_intervals = sorted(self.intervals, key=lambda x: x.start)
        for i in range(len(sorted_intervals) - 1):
            curr_act = sorted_intervals[i].activity
            next_act = sorted_intervals[i + 1].activity
            if curr_act != next_act:
                transitions.append({
                    "from_activity": curr_act,
                    "to_activity": next_act,
                    "transition_time": sorted_intervals[i].end,
                    "gap_sec": round(sorted_intervals[i + 1].start - sorted_intervals[i].end, 1)
                })
        return transitions

    def to_dict(self) -> Dict[str, Any]:
        """Serializes timeline to JSON-friendly dictionary."""
        return {
            "session_id": self.session_id,
            "session_start": round(self.session_start, 1),
            "session_end": round(self.session_end, 1),
            "total_duration_sec": round(self.total_duration, 1),
            "total_intervals": len(self.intervals),
            "intervals": [inv.to_dict() for inv in self.intervals],
            "transitions": self.get_transitions()
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)
