"""
Temporal timelines for ExtraSensory context and place labels.

These are categorical annotations, not raw GPS coordinates. Missing labels are
kept distinct from explicit negative labels so absence is never treated as proof.
"""

from typing import Dict, List, Optional, Type, TypeVar

import numpy as np


class LabelInterval:
    def __init__(self, label: str, start_time: float, end_time: float):
        self.label = label
        self.start = float(start_time)
        self.end = float(end_time)
        self.duration = max(0.0, self.end - self.start)

    def to_dict(self) -> Dict[str, object]:
        return {
            "label": self.label,
            "start": round(self.start, 1),
            "end": round(self.end, 1),
            "duration": round(self.duration, 1),
        }


TLabelTimeline = TypeVar("TLabelTimeline", bound="LabelTimeline")


class LabelTimeline:
    source_name = "ExtraSensory labels"

    def __init__(self, session_id: str = "session_001"):
        self.session_id = session_id
        self.session_end = 0.0
        self.window_sec = 60.0
        self.intervals: Dict[str, List[LabelInterval]] = {}
        self.observed_intervals: Dict[str, List[LabelInterval]] = {}

    @classmethod
    def from_labels(
        cls: Type[TLabelTimeline],
        timestamps: np.ndarray,
        label_values: Dict[str, np.ndarray],
        session_id: str = "session_001",
        time_origin: Optional[float] = None,
        session_end: Optional[float] = None,
        window_sec: Optional[float] = None,
    ) -> TLabelTimeline:
        timeline = cls(session_id=session_id)
        if len(timestamps) == 0:
            return timeline

        timestamps = np.asarray(timestamps, dtype=float)
        order = np.argsort(timestamps)
        sorted_timestamps = timestamps[order]
        positive_diffs = np.diff(sorted_timestamps)
        positive_diffs = positive_diffs[positive_diffs > 0]
        inferred_window = float(np.median(positive_diffs)) if len(positive_diffs) else 60.0
        timeline.window_sec = window_sec or inferred_window

        origin = float(time_origin) if time_origin is not None else float(sorted_timestamps[0])
        relative_timestamps = sorted_timestamps - origin
        inferred_end = max(0.0, float(relative_timestamps[-1] + timeline.window_sec))
        timeline.session_end = max(0.0, session_end) if session_end is not None else inferred_end

        for label, raw_values in label_values.items():
            values = np.asarray(raw_values, dtype=float)
            if len(values) != len(timestamps):
                raise ValueError(
                    f"Label '{label}' has {len(values)} values for {len(timestamps)} timestamps."
                )
            values = values[order]
            observed = np.isfinite(values)
            positive = observed & np.isclose(values, 1.0)
            timeline.observed_intervals[label] = timeline._build_intervals(
                label, relative_timestamps, observed
            )
            timeline.intervals[label] = timeline._build_intervals(
                label, relative_timestamps, positive
            )

        return timeline

    def _build_intervals(
        self, label: str, timestamps: np.ndarray, selected: np.ndarray
    ) -> List[LabelInterval]:
        indices = np.flatnonzero(selected)
        if len(indices) == 0:
            return []

        intervals: List[LabelInterval] = []
        group_start = previous_index = int(indices[0])
        for index_value in indices[1:]:
            index = int(index_value)
            gap = float(timestamps[index] - timestamps[previous_index])
            if gap > self.window_sec * 1.5:
                intervals.append(self._make_interval(label, timestamps, group_start, previous_index))
                group_start = index
            previous_index = index

        intervals.append(self._make_interval(label, timestamps, group_start, previous_index))
        return intervals

    def _make_interval(
        self, label: str, timestamps: np.ndarray, start_index: int, end_index: int
    ) -> LabelInterval:
        start = max(0.0, float(timestamps[start_index]))
        end = min(self.session_end, float(timestamps[end_index] + self.window_sec))
        return LabelInterval(label, start, max(start, end))

    def get_intervals(
        self,
        label: str,
        t_min: Optional[float] = None,
        t_max: Optional[float] = None,
    ) -> List[LabelInterval]:
        result = []
        for interval in self.intervals.get(label, []):
            start = max(interval.start, t_min) if t_min is not None else interval.start
            end = min(interval.end, t_max) if t_max is not None else interval.end
            if end > start:
                result.append(LabelInterval(label, start, end))
        return result

    def observation_fraction(
        self,
        label: str,
        t_min: Optional[float] = None,
        t_max: Optional[float] = None,
    ) -> float:
        start = max(0.0, t_min) if t_min is not None else 0.0
        end = min(self.session_end, t_max) if t_max is not None else self.session_end
        requested_duration = max(0.0, end - start)
        if requested_duration == 0:
            return 0.0

        observed_duration = sum(
            max(
                0.0,
                min(interval.end, end) - max(interval.start, start),
            )
            for interval in self.observed_intervals.get(label, [])
        )
        return min(1.0, observed_duration / requested_duration)

    def is_label_available(self, label: str) -> bool:
        return bool(self.observed_intervals.get(label))

    def get_transitions(
        self,
        from_label: str,
        to_label: str,
        t_min: Optional[float] = None,
        t_max: Optional[float] = None,
    ) -> List[Dict[str, object]]:
        ordered = sorted(
            (interval for intervals in self.intervals.values() for interval in intervals),
            key=lambda interval: (interval.start, interval.end),
        )
        transitions = []
        for current, following in zip(ordered, ordered[1:]):
            gap = following.start - current.end
            if (
                current.label == from_label
                and following.label == to_label
                and 0 <= gap <= self.window_sec * 1.5
                and (t_min is None or following.start >= t_min)
                and (t_max is None or following.start <= t_max)
            ):
                transitions.append(
                    {
                        "from_label": from_label,
                        "to_label": to_label,
                        "transition_time": round(following.start, 1),
                        "unobserved_gap_sec": round(gap, 1),
                    }
                )
        return transitions

    def to_dict(self) -> Dict[str, object]:
        labels = sorted(set(self.intervals) | set(self.observed_intervals))
        return {
            "session_id": self.session_id,
            "source": self.source_name,
            "session_duration_sec": round(self.session_end, 1),
            "labels": {
                label: {
                    "intervals": [item.to_dict() for item in self.intervals.get(label, [])],
                    "observed_duration_sec": round(
                        sum(item.duration for item in self.observed_intervals.get(label, [])), 1
                    ),
                    "observation_fraction": round(
                        self.observation_fraction(label), 4
                    ),
                }
                for label in labels
            },
        }


class ContextTimeline(LabelTimeline):
    source_name = "ExtraSensory context labels"


class LocationTimeline(LabelTimeline):
    source_name = "ExtraSensory place labels (coarse; no raw coordinates)"
