"""
deterministic_engine.py
=======================
Deterministic Computation Engine (JARVIS paradigm).
Executes formal mathematical, statistical, and temporal logic directly over
the Activity Timeline evidence store without allowing LLM arithmetic hallucinations.

Operations supported:
  - Total duration (SUM)
  - Interval count (COUNT)
  - Activity identification at timestamp / interval (ARGMAX)
  - Binary activity verification (EXISTS)
  - Onset & Offset timing (FIRST / LAST)
  - Comparative analysis (DIFF / RATIO)
  - State transition counting (TRANSITION_COUNT)
  - Open-world semantic reasoning aggregation
  - Intersection over Union (IoU) temporal grounding evaluation
"""

from typing import Dict, List, Any, Optional, Tuple
import numpy as np

from src.timeline_store import ActivityTimeline, ActivityInterval
from src.query_planner import QueryPlan
from src.dataset_loader import TARGET_ACTIVITIES, ACTIVITY_DISPLAY_NAMES, ACTIVITY_MODALITIES, DEFAULT_CHANNELS
from src.context_store import ContextTimeline, LabelInterval, LocationTimeline


class EvidencePackage:
    """
    Structured package containing all retrieved sensor evidence and exact deterministic computation.
    """
    def __init__(
        self,
        operation: str,
        result_value: Any,
        activity_name: str,
        intervals: List[ActivityInterval],
        modality: List[str],
        channels: List[str],
        calculation_trace: str,
        is_supported: bool = True,
        unsupported_reason: Optional[str] = None,
        confidence: float = 1.0,
        metadata: Optional[Dict[str, Any]] = None
    ):
        self.operation = operation
        self.result_value = result_value
        self.activity_name = activity_name
        self.intervals = intervals
        self.modality = modality
        self.channels = channels
        self.calculation_trace = calculation_trace
        self.is_supported = is_supported
        self.unsupported_reason = unsupported_reason
        self.confidence = float(confidence)
        self.metadata = metadata or {}

    def get_timestamp_strings(self) -> List[str]:
        """Returns formatted timestamp string representations (e.g. '905–1420 s')."""
        if not self.intervals:
            return ["None detected"]
        return [f"{int(inv.start)}–{int(inv.end)} s" for inv in self.intervals]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "operation": self.operation,
            "result_value": self.result_value,
            "activity_name": self.activity_name,
            "timestamps": self.get_timestamp_strings(),
            "intervals": [inv.to_dict() for inv in self.intervals],
            "modality": self.modality,
            "channels": self.channels,
            "calculation_trace": self.calculation_trace,
            "confidence": round(self.confidence, 4),
            "is_supported": self.is_supported,
            "unsupported_reason": self.unsupported_reason,
            "metadata": self.metadata
        }


class DeterministicEngine:
    """
    Deterministic Query Executor.
    """
    def __init__(self):
        pass

    def execute(
        self,
        plan: QueryPlan,
        timeline: ActivityTimeline,
        context_timeline: Optional[ContextTimeline] = None,
        location_timeline: Optional[LocationTimeline] = None,
    ) -> EvidencePackage:
        """
        Executes a QueryPlan deterministically against an ActivityTimeline.
        """
        # 1. Handle Unsupported Queries (DrHouse truthful rejection)
        if not plan.is_supported:
            return EvidencePackage(
                operation="unsupported",
                result_value=None,
                activity_name="N/A",
                intervals=[],
                modality=[],
                channels=[],
                calculation_trace=(
                    "Query rejected: requested information is unavailable from the "
                    "provided ExtraSensory labels and sensor fields."
                ),
                is_supported=False,
                unsupported_reason=plan.unsupported_reason,
                confidence=0.0
            )

        if plan.operation in {"context_filter", "co_occurrence", "context_presence"}:
            label = plan.context_label or plan.location_label
            label_timeline = (
                context_timeline if plan.context_label else location_timeline
            )
            if label is None or label_timeline is None or not label_timeline.is_label_available(label):
                return self._unavailable_label_evidence(
                    plan,
                    label or "requested context",
                    "No observations for this context/place label are available in this session.",
                )

            label_intervals = label_timeline.get_intervals(
                label, t_min=plan.t_min, t_max=plan.t_max
            )
            coverage = label_timeline.observation_fraction(
                label, t_min=plan.t_min, t_max=plan.t_max
            )
            if plan.target_activity:
                activity_intervals = timeline.get_intervals_for_activity(
                    plan.target_activity, t_min=plan.t_min, t_max=plan.t_max
                )
                matched_intervals = self._intersect_activity_labels(
                    plan.target_activity, activity_intervals, label_intervals
                )
                activity_name = (
                    f"{ACTIVITY_DISPLAY_NAMES.get(plan.target_activity, plan.target_activity)} "
                    f"while {label.replace('_', ' ')}"
                )
            else:
                matched_intervals = [
                    ActivityInterval(
                        label, item.start, item.end, coverage,
                        modality=[label_timeline.source_name], channels=[]
                    )
                    for item in label_intervals
                ]
                activity_name = label.replace("_", " ").capitalize()

            duration = sum(item.duration for item in matched_intervals)
            partial_note = (
                f"; label observations cover {coverage:.1%} of the requested window"
                if coverage < 0.99 else ""
            )
            if plan.operation == "co_occurrence":
                occurred = bool(matched_intervals)
                result = "Yes" if occurred else ("No" if coverage >= 0.99 else "Unknown")
                trace = (
                    f"EXISTS({activity_name}) = {result}; observed overlap {duration:.1f} s"
                    f"{partial_note}"
                )
                metadata = {
                    "occurred": occurred,
                    "label": label,
                    "label_source": label_timeline.source_name,
                    "observation_fraction": coverage,
                }
            elif plan.operation == "context_presence":
                occurred = bool(matched_intervals)
                result = "Yes" if occurred else ("No" if coverage >= 0.99 else "Unknown")
                trace = (
                    f"EXISTS({label.replace('_', ' ')}) = {result}"
                    f"{partial_note}"
                )
                metadata = {
                    "occurred": occurred,
                    "label": label,
                    "label_source": label_timeline.source_name,
                    "observation_fraction": coverage,
                }
            elif plan.query_metric == "count":
                result = len(matched_intervals)
                trace = (
                    f"COUNT({activity_name} labeled intervals) = {result}"
                    f"{partial_note}"
                )
                metadata = {
                    "count": result,
                    "label": label,
                    "label_source": label_timeline.source_name,
                    "observation_fraction": coverage,
                }
            else:
                result = round(duration, 1)
                trace = (
                    f"SUM({activity_name} labeled overlap) = {duration:.1f} s"
                    f"{partial_note}"
                )
                metadata = {
                    "total_seconds": result,
                    "label": label,
                    "label_source": label_timeline.source_name,
                    "observation_fraction": coverage,
                }

            modalities = (
                ["accelerometer", "gyroscope", label_timeline.source_name]
                if plan.target_activity
                else [label_timeline.source_name]
            )
            channels = DEFAULT_CHANNELS if plan.target_activity else []
            return EvidencePackage(
                operation=plan.operation,
                result_value=result,
                activity_name=activity_name,
                intervals=matched_intervals,
                modality=modalities,
                channels=channels,
                calculation_trace=trace,
                confidence=coverage,
                metadata=metadata,
            )

        if plan.operation in {
            "location_duration", "location_count", "location_identification"
        }:
            if location_timeline is None:
                return self._unavailable_label_evidence(
                    plan,
                    plan.location_label or "place",
                    "This session has no place-label timeline.",
                )
            if plan.location_label and not location_timeline.is_label_available(plan.location_label):
                return self._unavailable_label_evidence(
                    plan,
                    plan.location_label,
                    "No observations for this place label are available in this session.",
                )

            if plan.location_label:
                labels = [plan.location_label]
            else:
                labels = sorted(location_timeline.intervals)
                if not any(location_timeline.is_label_available(label) for label in labels):
                    return self._unavailable_label_evidence(
                        plan,
                        "place",
                        "No place-label observations are available in this session.",
                    )
            matching_intervals = []
            coverages = []
            coverage_t_min = plan.t_min
            coverage_t_max = plan.t_max
            if plan.operation == "location_identification" and plan.target_timestamp is not None:
                coverage_t_min = plan.target_timestamp
                coverage_t_max = min(
                    location_timeline.session_end,
                    plan.target_timestamp + 1.0,
                )
            for label in labels:
                matching_intervals.extend(
                    location_timeline.get_intervals(label, plan.t_min, plan.t_max)
                )
                coverages.append(
                    location_timeline.observation_fraction(
                        label, coverage_t_min, coverage_t_max
                    )
                    if location_timeline.is_label_available(label)
                    else 0.0
                )
            matching_intervals.sort(key=lambda item: item.start)
            coverage = min(coverages) if coverages else 0.0

            if plan.operation == "location_identification":
                if plan.target_timestamp is not None:
                    matching_intervals = [
                        item for item in matching_intervals
                        if item.start <= plan.target_timestamp < item.end
                    ]
                result = sorted({item.label for item in matching_intervals})
                activity_name = ", ".join(label.replace("_", " ") for label in result) or "Place label"
                trace = (
                    f"Observed place labels: {', '.join(result) if result else 'none'}; "
                                       f"label observation fraction {coverage:.1%}. "
                    "No raw coordinates are available."
                )
                operation = "location_identification"
            elif plan.operation == "location_count":
                result = len(matching_intervals)
                activity_name = plan.location_label.replace("_", " ").capitalize()
                trace = (
                    f"COUNT({activity_name} labeled visits) = {result}; "
                    f"label observation fraction {coverage:.1%}"
                )
                operation = "location_count"
            else:
                result = round(sum(item.duration for item in matching_intervals), 1)
                activity_name = plan.location_label.replace("_", " ").capitalize()
                trace = (
                    f"SUM({activity_name} labeled intervals) = {result:.1f} s; "
                    f"label observation fraction {coverage:.1%}"
                )
                operation = "location_duration"

            evidence_intervals = [
                ActivityInterval(
                    item.label, item.start, item.end, coverage,
                    modality=[location_timeline.source_name], channels=[]
                )
                for item in matching_intervals
            ]
            return EvidencePackage(
                operation=operation,
                result_value=result,
                activity_name=activity_name,
                intervals=evidence_intervals,
                modality=[location_timeline.source_name],
                channels=[],
                calculation_trace=trace,
                confidence=coverage,
                metadata={
                    "location_label": plan.location_label,
                    "observation_fraction": coverage,
                    "raw_coordinates_available": False,
                },
            )

        if plan.operation == "place_transition":
            if location_timeline is None:
                return self._unavailable_label_evidence(
                    plan, "place transitions", "This session has no place-label timeline."
                )
            from_label = plan.from_location or ""
            to_label = plan.to_location or ""
            if not all(
                location_timeline.is_label_available(label)
                for label in (from_label, to_label)
            ):
                return self._unavailable_label_evidence(
                    plan,
                    f"{from_label} to {to_label}",
                    "One or both place labels have no observations in this session.",
                )
            transitions = location_timeline.get_transitions(
                from_label, to_label, plan.t_min, plan.t_max
            )
            coverage = min(
                location_timeline.observation_fraction(from_label, plan.t_min, plan.t_max),
                location_timeline.observation_fraction(to_label, plan.t_min, plan.t_max),
            )
            evidence_intervals = [
                ActivityInterval(
                    label, item.start, item.end, coverage,
                    modality=[location_timeline.source_name], channels=[]
                )
                for label in dict.fromkeys((from_label, to_label))
                for item in location_timeline.get_intervals(
                    label, t_min=plan.t_min, t_max=plan.t_max
                )
            ]
            evidence_intervals.sort(key=lambda item: item.start)
            return EvidencePackage(
                operation="place_transition",
                result_value=len(transitions),
                activity_name=f"{from_label.capitalize()} to {to_label.capitalize()}",
                intervals=evidence_intervals,
                modality=[location_timeline.source_name],
                channels=[],
                calculation_trace=(
                    f"COUNT(Observed transition {from_label} -> {to_label}) = "
                    f"{len(transitions)}; label observation fraction {coverage:.1%}"
                ),
                confidence=coverage,
                metadata={
                    "transitions": transitions,
                    "observation_fraction": coverage,
                    "raw_coordinates_available": False,
                },
            )

        # 2. Duration Query (SUM)
        if plan.operation == "duration":
            act = plan.target_activity or "walking"
            matching_intervals = timeline.get_intervals_for_activity(act, t_min=plan.t_min, t_max=plan.t_max)
            durations = [inv.duration for inv in matching_intervals]
            total_duration = sum(durations)
            avg_conf = np.mean([inv.confidence for inv in matching_intervals]) if matching_intervals else 1.0

            trace_parts = [f"{int(inv.end)} - {int(inv.start)} = {int(inv.duration)} s" for inv in matching_intervals]
            trace_str = " + ".join(trace_parts) + f" = Total {int(total_duration)} s" if trace_parts else "No intervals detected (0 s)"

            return EvidencePackage(
                operation="duration",
                result_value=round(total_duration, 1),
                activity_name=ACTIVITY_DISPLAY_NAMES.get(act, act),
                intervals=matching_intervals,
                modality=ACTIVITY_MODALITIES.get(act, ["accelerometer", "gyroscope"]),
                channels=DEFAULT_CHANNELS,
                calculation_trace=trace_str,
                confidence=avg_conf,
                metadata={"total_seconds": round(total_duration, 1), "minutes": round(total_duration / 60.0, 2)}
            )

        # 3. Count / Frequency Query (COUNT)
        if plan.operation == "count":
            act = plan.target_activity or "running"
            matching_intervals = timeline.get_intervals_for_activity(act, t_min=plan.t_min, t_max=plan.t_max)
            count = len(matching_intervals)
            avg_conf = np.mean([inv.confidence for inv in matching_intervals]) if matching_intervals else 1.0
            trace_str = f"COUNT({ACTIVITY_DISPLAY_NAMES.get(act, act)} intervals) = {count}"

            return EvidencePackage(
                operation="count",
                result_value=count,
                activity_name=ACTIVITY_DISPLAY_NAMES.get(act, act),
                intervals=matching_intervals,
                modality=ACTIVITY_MODALITIES.get(act, ["accelerometer", "gyroscope"]),
                channels=DEFAULT_CHANNELS,
                calculation_trace=trace_str,
                confidence=avg_conf,
                metadata={"count": count}
            )

        # 4. Onset Query (FIRST)
        if plan.operation == "onset":
            act = plan.target_activity or "running"
            matching_intervals = timeline.get_intervals_for_activity(act, t_min=plan.t_min, t_max=plan.t_max)
            if matching_intervals:
                first_inv = matching_intervals[0]
                onset_time = first_inv.start
                trace_str = f"FIRST({ACTIVITY_DISPLAY_NAMES.get(act, act)}.start) = {int(onset_time)} s"
                return EvidencePackage(
                    operation="onset",
                    result_value=round(onset_time, 1),
                    activity_name=ACTIVITY_DISPLAY_NAMES.get(act, act),
                    intervals=[first_inv],
                    modality=ACTIVITY_MODALITIES.get(act, ["accelerometer", "gyroscope"]),
                    channels=DEFAULT_CHANNELS,
                    calculation_trace=trace_str,
                    confidence=first_inv.confidence,
                    metadata={"onset_sec": round(onset_time, 1)}
                )
            else:
                return EvidencePackage(
                    operation="onset",
                    result_value=None,
                    activity_name=ACTIVITY_DISPLAY_NAMES.get(act, act),
                    intervals=[],
                    modality=ACTIVITY_MODALITIES.get(act, ["accelerometer", "gyroscope"]),
                    channels=DEFAULT_CHANNELS,
                    calculation_trace=f"No {ACTIVITY_DISPLAY_NAMES.get(act, act)} activity was detected in the specified recording window.",
                    confidence=1.0
                )

        # 5. Offset Query (LAST)
        if plan.operation == "offset":
            act = plan.target_activity or "walking"
            matching_intervals = timeline.get_intervals_for_activity(act, t_min=plan.t_min, t_max=plan.t_max)
            if matching_intervals:
                last_inv = matching_intervals[-1]
                offset_time = last_inv.end
                trace_str = f"LAST({ACTIVITY_DISPLAY_NAMES.get(act, act)}.end) = {int(offset_time)} s"
                return EvidencePackage(
                    operation="offset",
                    result_value=round(offset_time, 1),
                    activity_name=ACTIVITY_DISPLAY_NAMES.get(act, act),
                    intervals=[last_inv],
                    modality=ACTIVITY_MODALITIES.get(act, ["accelerometer", "gyroscope"]),
                    channels=DEFAULT_CHANNELS,
                    calculation_trace=trace_str,
                    confidence=last_inv.confidence,
                    metadata={"offset_sec": round(offset_time, 1)}
                )
            else:
                return EvidencePackage(
                    operation="offset",
                    result_value=None,
                    activity_name=ACTIVITY_DISPLAY_NAMES.get(act, act),
                    intervals=[],
                    modality=ACTIVITY_MODALITIES.get(act, ["accelerometer", "gyroscope"]),
                    channels=DEFAULT_CHANNELS,
                    calculation_trace=f"No {ACTIVITY_DISPLAY_NAMES.get(act, act)} activity was detected in the specified recording window.",
                    confidence=1.0
                )

        # 6. Comparison Query (DIFF / MAX)
        if plan.operation == "comparison":
            act1 = plan.target_activity or "walking"
            act2 = plan.compare_activity or "sitting"
            inv1 = timeline.get_intervals_for_activity(act1, t_min=plan.t_min, t_max=plan.t_max)
            inv2 = timeline.get_intervals_for_activity(act2, t_min=plan.t_min, t_max=plan.t_max)
            dur1 = sum(i.duration for i in inv1)
            dur2 = sum(i.duration for i in inv2)

            name1 = ACTIVITY_DISPLAY_NAMES.get(act1, act1)
            name2 = ACTIVITY_DISPLAY_NAMES.get(act2, act2)

            if dur1 > dur2:
                winner = name1
                diff = dur1 - dur2
                trace_str = f"{name1} ({int(dur1)} s) > {name2} ({int(dur2)} s) by {int(diff)} s"
            elif dur2 > dur1:
                winner = name2
                diff = dur2 - dur1
                trace_str = f"{name2} ({int(dur2)} s) > {name1} ({int(dur1)} s) by {int(diff)} s"
            else:
                winner = "Equal"
                diff = 0
                trace_str = f"{name1} ({int(dur1)} s) == {name2} ({int(dur2)} s)"

            all_inv = inv1 + inv2
            avg_conf = np.mean([i.confidence for i in all_inv]) if all_inv else 1.0

            return EvidencePackage(
                operation="comparison",
                result_value=winner,
                activity_name=f"{name1} vs {name2}",
                intervals=all_inv,
                modality=["accelerometer", "gyroscope"],
                channels=DEFAULT_CHANNELS,
                calculation_trace=trace_str,
                confidence=avg_conf,
                metadata={"dur1": dur1, "dur2": dur2, "diff": diff, "winner": winner}
            )

        # 7. Transition Query
        if plan.operation == "transition":
            from_act = plan.from_activity or "sitting"
            to_act = plan.to_activity or "walking"
            from_name = ACTIVITY_DISPLAY_NAMES.get(from_act, from_act)
            to_name = ACTIVITY_DISPLAY_NAMES.get(to_act, to_act)

            transitions = timeline.get_transitions()
            matching_trans = [t for t in transitions if t["from_activity"] == from_act and t["to_activity"] == to_act]
            
            # Filter by time if given
            if plan.t_min is not None or plan.t_max is not None:
                matching_trans = [
                    t for t in matching_trans
                    if (plan.t_min is None or t["transition_time"] >= plan.t_min) and
                       (plan.t_max is None or t["transition_time"] <= plan.t_max)
                ]

            count = len(matching_trans)
            trans_times = [f"{int(t['transition_time'])} s" for t in matching_trans]
            trace_str = f"COUNT(Transition {from_name} -> {to_name}) = {count}" + (f" at [{', '.join(trans_times)}]" if trans_times else "")

            # Associated intervals
            inv_from = timeline.get_intervals_for_activity(from_act, t_min=plan.t_min, t_max=plan.t_max)
            inv_to = timeline.get_intervals_for_activity(to_act, t_min=plan.t_min, t_max=plan.t_max)

            return EvidencePackage(
                operation="transition",
                result_value=count,
                activity_name=f"Transition: {from_name} to {to_name}",
                intervals=inv_from + inv_to,
                modality=["accelerometer", "gyroscope"],
                channels=DEFAULT_CHANNELS,
                calculation_trace=trace_str,
                confidence=0.95 if count > 0 else 1.0,
                metadata={"transition_count": count, "transitions": matching_trans}
            )

        # 8. Open-World Semantic Reasoning Query
        if plan.operation == "open_world_reasoning":
            target_acts = plan.target_activities_group or ["bicycling"]
            category_name = plan.semantic_category or "Semantic Category"
            
            all_matching_inv = []
            for act in target_acts:
                invs = timeline.get_intervals_for_activity(act, t_min=plan.t_min, t_max=plan.t_max)
                all_matching_inv.extend(invs)

            total_dur = sum(i.duration for i in all_matching_inv)
            occurred = len(all_matching_inv) > 0
            act_names = [ACTIVITY_DISPLAY_NAMES.get(a, a) for a in target_acts]

            trace_str = f"Category '{category_name}' mapped to [{', '.join(act_names)}]. Detected {len(all_matching_inv)} intervals totaling {int(total_dur)} s."

            return EvidencePackage(
                operation="open_world_reasoning",
                result_value="Yes" if occurred else "No",
                activity_name=", ".join(act_names),
                intervals=all_matching_inv,
                modality=["accelerometer", "gyroscope"],
                channels=DEFAULT_CHANNELS,
                calculation_trace=trace_str,
                confidence=np.mean([i.confidence for i in all_matching_inv]) if all_matching_inv else 1.0,
                metadata={"occurred": occurred, "total_duration": total_dur, "category": category_name}
            )

        # 9. Verification Query ("Is the user running?")
        if plan.operation == "verification":
            act = plan.target_activity or "running"
            matching_inv = timeline.get_intervals_for_activity(act, t_min=plan.t_min, t_max=plan.t_max)
            act_name = ACTIVITY_DISPLAY_NAMES.get(act, act)
            is_active = len(matching_inv) > 0
            res_str = "Yes" if is_active else "No"
            trace_str = f"EXISTS({act_name}) in window = {is_active} ({len(matching_inv)} intervals found)"

            return EvidencePackage(
                operation="verification",
                result_value=res_str,
                activity_name=act_name,
                intervals=matching_inv,
                modality=ACTIVITY_MODALITIES.get(act, ["accelerometer", "gyroscope"]),
                channels=DEFAULT_CHANNELS,
                calculation_trace=trace_str,
                confidence=np.mean([i.confidence for i in matching_inv]) if matching_inv else 1.0,
                metadata={"is_active": is_active}
            )

        # 10. General Identification Query ("What was the user doing?")
        all_inv = timeline.get_all_intervals(t_min=plan.t_min, t_max=plan.t_max)
        if not all_inv:
            return EvidencePackage(
                operation="identification",
                result_value="No activity detected",
                activity_name="None",
                intervals=[],
                modality=["accelerometer", "gyroscope"],
                channels=DEFAULT_CHANNELS,
                calculation_trace="No activity intervals found within temporal bounds.",
                confidence=1.0
            )

        # Rank by total duration within window
        act_durations: Dict[str, float] = {}
        for inv in all_inv:
            act_durations[inv.activity] = act_durations.get(inv.activity, 0.0) + inv.duration

        dominant_act = max(act_durations.keys(), key=lambda k: act_durations[k])
        dominant_name = ACTIVITY_DISPLAY_NAMES.get(dominant_act, dominant_act)
        trace_str = f"Dominant activity: {dominant_name} ({int(act_durations[dominant_act])} s)"

        return EvidencePackage(
            operation="identification",
            result_value=dominant_name,
            activity_name=dominant_name,
            intervals=[inv for inv in all_inv if inv.activity == dominant_act],
            modality=ACTIVITY_MODALITIES.get(dominant_act, ["accelerometer", "gyroscope"]),
            channels=DEFAULT_CHANNELS,
            calculation_trace=trace_str,
            confidence=np.mean([i.confidence for i in all_inv if i.activity == dominant_act]),
            metadata={"dominant_activity": dominant_act, "durations": act_durations}
        )

    @staticmethod
    def _intersect_activity_labels(
        activity: str,
        activity_intervals: List[ActivityInterval],
        label_intervals: List[LabelInterval],
    ) -> List[ActivityInterval]:
        intersections = []
        for activity_interval in activity_intervals:
            for label_interval in label_intervals:
                start = max(activity_interval.start, label_interval.start)
                end = min(activity_interval.end, label_interval.end)
                if end > start:
                    intersections.append(
                        ActivityInterval(
                            activity,
                            start,
                            end,
                            activity_interval.confidence,
                            modality=activity_interval.modality,
                            channels=activity_interval.channels,
                        )
                    )
        return sorted(intersections, key=lambda item: item.start)

    @staticmethod
    def _unavailable_label_evidence(
        plan: QueryPlan, label: str, reason: str
    ) -> EvidencePackage:
        return EvidencePackage(
            operation=plan.operation,
            result_value=None,
            activity_name=label.replace("_", " ").capitalize(),
            intervals=[],
            modality=[],
            channels=[],
            calculation_trace=reason,
            is_supported=False,
            unsupported_reason=reason,
            confidence=0.0,
            metadata={"label": label, "raw_coordinates_available": False},
        )

    def compute_temporal_iou(self, predicted_intervals: List[Tuple[float, float]], ground_truth_intervals: List[Tuple[float, float]]) -> float:
        """
        Computes exact Intersection over Union (IoU) metric for temporal evidence grounding.
        """
        if not predicted_intervals and not ground_truth_intervals:
            return 1.0
        if not predicted_intervals or not ground_truth_intervals:
            return 0.0

        # Discretize timeline at 1-second resolution
        t_max = max(max(p[1] for p in predicted_intervals), max(g[1] for g in ground_truth_intervals))
        t_min = min(min(p[0] for p in predicted_intervals), min(g[0] for g in ground_truth_intervals))
        length = int(t_max - t_min) + 1
        if length <= 0:
            return 0.0

        pred_mask = np.zeros(length, dtype=bool)
        gt_mask = np.zeros(length, dtype=bool)

        for s, e in predicted_intervals:
            idx_s = max(0, int(s - t_min))
            idx_e = min(length, int(e - t_min))
            pred_mask[idx_s:idx_e] = True

        for s, e in ground_truth_intervals:
            idx_s = max(0, int(s - t_min))
            idx_e = min(length, int(e - t_min))
            gt_mask[idx_s:idx_e] = True

        intersection = np.sum(pred_mask & gt_mask)
        union = np.sum(pred_mask | gt_mask)

        return float(intersection / union) if union > 0 else 0.0
