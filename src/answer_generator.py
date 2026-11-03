"""
answer_generator.py
===================
Answer Assembly and Grounded Explanation Generator.
Produces structured, factual responses adhering strictly to the Challenge schema:
  - Answer
  - Activity/Event
  - Evidence
  - Timestamp(s)
  - Sensor Modality
  - Sensor Channel(s)
  - Explanation
"""

from typing import Dict, List, Any, Optional
from src.deterministic_engine import EvidencePackage
from src.query_planner import QueryPlan
from src.nim_llm import NvidiaNimClient


class StructuredAnswer:
    def __init__(
        self,
        answer: str,
        activity_event: str,
        evidence: str,
        timestamps: str,
        sensor_modality: str,
        sensor_channels: str,
        explanation: str,
        confidence: float,
        is_supported: bool = True,
        raw_evidence: Optional[Dict[str, Any]] = None
    ):
        self.answer = answer
        self.activity_event = activity_event
        self.evidence = evidence
        self.timestamps = timestamps
        self.sensor_modality = sensor_modality
        self.sensor_channels = sensor_channels
        self.explanation = explanation
        self.confidence = float(confidence)
        self.is_supported = is_supported
        self.raw_evidence = raw_evidence or {}

    def format_text(self) -> str:
        """Formats the response in the canonical Challenge 1 text schema."""
        return (
            f"Answer: {self.answer}\n\n"
            f"Activity/Event: {self.activity_event}\n\n"
            f"Evidence: {self.evidence}\n"
            f"Timestamp(s): {self.timestamps}\n\n"
            f"Sensor Modality:\n{self.sensor_modality}\n\n"
            f"Sensor Channel(s):\n{self.sensor_channels}\n\n"
            f"Explanation:\n{self.explanation}"
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "answer": self.answer,
            "activity_event": self.activity_event,
            "evidence": self.evidence,
            "timestamps": self.timestamps,
            "sensor_modality": self.sensor_modality,
            "sensor_channels": self.sensor_channels,
            "explanation": self.explanation,
            "confidence": round(self.confidence, 4),
            "is_supported": self.is_supported,
            "raw_evidence": self.raw_evidence
        }


class AnswerGenerator:
    """
    Assembles evidence packages into rigorous grounded structured responses.
    Optionally uses NVIDIA NIM for activity-answer explanation. Context/place
    answers remain deterministic and do not send label data to an external model.
    """
    def __init__(self, nim_client: Optional[NvidiaNimClient] = None, use_nim: bool = True):
        self.nim_client = nim_client or (NvidiaNimClient() if use_nim else None)

    def generate(self, plan: QueryPlan, evidence: EvidencePackage) -> StructuredAnswer:
        """
        Builds a compliant structured answer from deterministic query execution.
        """
        # 1. Handle Unsupported / Out-of-Scope Queries (DrHouse truthful rejection)
        if not evidence.is_supported or not plan.is_supported:
            reason = (
                evidence.unsupported_reason
                or plan.unsupported_reason
                or "Requested information is unavailable from the provided ExtraSensory data."
            )
            return StructuredAnswer(
                answer="Unable to answer with available ExtraSensory data.",
                activity_event="N/A (Out-of-scope query)",
                evidence="None available.",
                timestamps="N/A",
                sensor_modality="Available ExtraSensory sensor and label data",
                sensor_channels="N/A",
                explanation=f"The question cannot be grounded in the available ExtraSensory data. {reason}",
                confidence=0.0,
                is_supported=False,
                raw_evidence=evidence.to_dict()
            )

        # Modality & Channels strings
        modality_str = " + ".join([m.capitalize() for m in evidence.modality]) if evidence.modality else "N/A"
        channels_str = " + ".join(evidence.channels) if evidence.channels else "N/A (no raw sensor channels)"
        ts_list = evidence.get_timestamp_strings()
        timestamps_str = ", ".join(ts_list) if ts_list else "None detected"

        if evidence.operation in {
            "context_filter", "co_occurrence", "context_presence",
            "location_duration", "location_count", "location_identification",
            "place_transition",
        }:
            metadata = evidence.metadata
            coverage = float(metadata.get("observation_fraction", 0.0))
            label = str(metadata.get("label") or metadata.get("location_label") or "")
            partial = coverage < 0.99
            answer: str
            if evidence.operation in {"co_occurrence", "context_presence"}:
                answer = str(evidence.result_value)
                if answer == "Unknown":
                    answer = "Unknown — label coverage is incomplete."
            elif evidence.operation in {"location_identification"}:
                places = evidence.result_value or []
                answer = (
                    "Observed coarse place labels: "
                    + ", ".join(str(place).replace("_", " ") for place in places)
                    if places else
                    ("Unknown — no place label was observed in the available data."
                     if partial else "No supported place label was recorded.")
                )
            elif evidence.operation == "place_transition":
                count = int(evidence.result_value or 0)
                answer = f"{'At least ' if partial else ''}{count} observed transitions"
            elif evidence.operation == "location_count" or (
                evidence.operation == "context_filter" and plan.query_metric == "count"
            ):
                count = int(evidence.result_value or 0)
                answer = f"{'At least ' if partial else ''}{count} observed interval(s)"
            else:
                duration = float(evidence.result_value or 0.0)
                answer = f"{'At least ' if partial else ''}{int(duration)} seconds"

            explanation = evidence.calculation_trace
            if partial:
                explanation += (
                    " Unlabeled time is unknown and is not counted as a negative observation."
                )
            if evidence.operation in {
                "location_duration", "location_count", "location_identification", "place_transition"
            }:
                explanation += (
                    " These are coarse ExtraSensory place annotations; raw GPS coordinates "
                    "and exact addresses are not available."
                )
            elif label:
                explanation += (
                    " This answer uses annotated context/place labels, not inferred or "
                    "unobserved context."
                )

            return StructuredAnswer(
                answer=answer,
                activity_event=evidence.activity_name,
                evidence=evidence.calculation_trace,
                timestamps=timestamps_str,
                sensor_modality=modality_str,
                sensor_channels=channels_str,
                explanation=explanation,
                confidence=evidence.confidence,
                raw_evidence=evidence.to_dict(),
            )

        # 2. Duration Query
        if evidence.operation == "duration":
            dur = evidence.result_value
            if dur is None or dur == 0 or not evidence.intervals:
                return StructuredAnswer(
                    answer="0 seconds",
                    activity_event=evidence.activity_name,
                    evidence=f"No {evidence.activity_name.lower()} activity was detected during the recording window.",
                    timestamps="None detected",
                    sensor_modality=modality_str,
                    sensor_channels=channels_str,
                    explanation=f"The sensor model evaluated all accelerometer and gyroscope streams across the recording timeline and detected zero intervals corresponding to {evidence.activity_name.lower()}.",
                    confidence=evidence.confidence,
                    raw_evidence=evidence.to_dict()
                )
            
            num_intervals = len(evidence.intervals)
            interval_str = f"{num_intervals} interval" if num_intervals == 1 else f"{num_intervals} distinct intervals"
            calc_text = evidence.calculation_trace
            
            return self._maybe_enrich_with_nim(plan, evidence, StructuredAnswer(
                answer=f"{int(dur)} seconds",
                activity_event=evidence.activity_name,
                evidence=f"{evidence.activity_name} was detected across {interval_str}.",
                timestamps=timestamps_str,
                sensor_modality=modality_str,
                sensor_channels=channels_str,
                explanation=f"{evidence.activity_name} was identified by characteristic inertial motion signatures during {interval_str}. Deterministic aggregation of the temporal bounds yields {calc_text}.",
                confidence=evidence.confidence,
                raw_evidence=evidence.to_dict()
            ))

        # 3. Count / Frequency Query
        if evidence.operation == "count":
            cnt = evidence.result_value
            plural = "times" if cnt != 1 else "time"
            return self._maybe_enrich_with_nim(plan, evidence, StructuredAnswer(
                answer=f"{cnt} {plural}",
                activity_event=evidence.activity_name,
                evidence=f"Detected {cnt} separate interval(s) of {evidence.activity_name.lower()}.",
                timestamps=timestamps_str,
                sensor_modality=modality_str,
                sensor_channels=channels_str,
                explanation=f"The activity timeline contains {cnt} continuous segments where accelerometer and gyroscope signals matched the kinematic pattern of {evidence.activity_name.lower()} with calibrated confidence >= 0.35.",
                confidence=evidence.confidence,
                raw_evidence=evidence.to_dict()
            ))

        # 4. Onset Query
        if evidence.operation == "onset":
            onset = evidence.result_value
            if onset is None:
                return StructuredAnswer(
                    answer="Not detected",
                    activity_event=evidence.activity_name,
                    evidence=f"No {evidence.activity_name.lower()} activity occurred in the recording.",
                    timestamps="None detected",
                    sensor_modality=modality_str,
                    sensor_channels=channels_str,
                    explanation=f"The sensor classifier found no episodes of {evidence.activity_name.lower()} within the analyzed timeline.",
                    confidence=evidence.confidence,
                    raw_evidence=evidence.to_dict()
                )
            return self._maybe_enrich_with_nim(plan, evidence, StructuredAnswer(
                answer=f"At {int(onset)} seconds",
                activity_event=evidence.activity_name,
                evidence=f"First onset of {evidence.activity_name.lower()} recorded at {int(onset)} s.",
                timestamps=timestamps_str,
                sensor_modality=modality_str,
                sensor_channels=channels_str,
                explanation=f"The initial onset for {evidence.activity_name.lower()} was identified at relative second {int(onset)}, supported by sustained kinematic shifts across the 6 inertial channels.",
                confidence=evidence.confidence,
                raw_evidence=evidence.to_dict()
            ))

        # 5. Offset Query
        if evidence.operation == "offset":
            offset = evidence.result_value
            if offset is None:
                return StructuredAnswer(
                    answer="Not detected",
                    activity_event=evidence.activity_name,
                    evidence=f"No {evidence.activity_name.lower()} activity occurred in the recording.",
                    timestamps="None detected",
                    sensor_modality=modality_str,
                    sensor_channels=channels_str,
                    explanation=f"The sensor classifier found no episodes of {evidence.activity_name.lower()} within the analyzed timeline.",
                    confidence=evidence.confidence,
                    raw_evidence=evidence.to_dict()
                )
            return self._maybe_enrich_with_nim(plan, evidence, StructuredAnswer(
                answer=f"At {int(offset)} seconds",
                activity_event=evidence.activity_name,
                evidence=f"Final offset of {evidence.activity_name.lower()} recorded at {int(offset)} s.",
                timestamps=timestamps_str,
                sensor_modality=modality_str,
                sensor_channels=channels_str,
                explanation=f"The final interval for {evidence.activity_name.lower()} concluded at second {int(offset)}, marking the cessation of that activity state.",
                confidence=evidence.confidence,
                raw_evidence=evidence.to_dict()
            ))

        # 6. Comparison Query
        if evidence.operation == "comparison":
            meta = evidence.metadata
            winner = evidence.result_value
            d1, d2 = meta.get("dur1", 0), meta.get("dur2", 0)
            diff = meta.get("diff", 0)
            return self._maybe_enrich_with_nim(plan, evidence, StructuredAnswer(
                answer=f"{winner} (by {int(diff)} seconds)",
                activity_event=evidence.activity_name,
                evidence=f"Comparative duration analysis: {evidence.calculation_trace}.",
                timestamps=timestamps_str,
                sensor_modality=modality_str,
                sensor_channels=channels_str,
                explanation=f"Deterministic temporal summation of sensor intervals reveals {evidence.calculation_trace}, confirming {winner} as the activity with greater duration.",
                confidence=evidence.confidence,
                raw_evidence=evidence.to_dict()
            ))

        # 7. Transition Query
        if evidence.operation == "transition":
            cnt = evidence.result_value
            plural = "times" if cnt != 1 else "time"
            return self._maybe_enrich_with_nim(plan, evidence, StructuredAnswer(
                answer=f"{cnt} {plural}",
                activity_event=evidence.activity_name,
                evidence=f"Identified {cnt} transition(s): {evidence.calculation_trace}.",
                timestamps=timestamps_str,
                sensor_modality=modality_str,
                sensor_channels=channels_str,
                explanation=f"Activity state transitions were detected by tracking consecutive non-overlapping interval boundaries where kinematic frequency and orientation shifted from {plan.from_activity} to {plan.to_activity}.",
                confidence=evidence.confidence,
                raw_evidence=evidence.to_dict()
            ))

        # 8. Open-World Semantic Reasoning Query
        if evidence.operation == "open_world_reasoning":
            meta = evidence.metadata
            occurred = meta.get("occurred", False)
            tot_dur = meta.get("total_duration", 0)
            cat_name = meta.get("category", "Target Category")
            
            if occurred:
                ans_str = f"Yes ({evidence.activity_name})"
                expl_str = f"Semantic reasoning maps '{cat_name}' to sensor activities [{evidence.activity_name}]. Sensor evidence confirms {len(evidence.intervals)} interval(s) totaling {int(tot_dur)} seconds of {evidence.activity_name.lower()}."
            else:
                ans_str = "No"
                expl_str = f"Semantic reasoning maps '{cat_name}' to sensor activities [{evidence.activity_name}]. Analysis of all accelerometer and gyroscope channels revealed no matching activity signatures during the recording."

            return self._maybe_enrich_with_nim(plan, evidence, StructuredAnswer(
                answer=ans_str,
                activity_event=f"{cat_name} ({evidence.activity_name})",
                evidence=evidence.calculation_trace,
                timestamps=timestamps_str,
                sensor_modality=modality_str,
                sensor_channels=channels_str,
                explanation=expl_str,
                confidence=evidence.confidence,
                raw_evidence=evidence.to_dict()
            ))

        # 9. Verification Query ("Is the user running?")
        if evidence.operation == "verification":
            is_active = evidence.metadata.get("is_active", False)
            ans_str = "Yes" if is_active else "No"
            expl_str = f"The sensor pipeline evaluated the 6-axis kinematic features. {evidence.activity_name} {'was confirmed' if is_active else 'was not observed'} in the analyzed time window."

            return self._maybe_enrich_with_nim(plan, evidence, StructuredAnswer(
                answer=ans_str,
                activity_event=evidence.activity_name,
                evidence=f"Verification check for {evidence.activity_name.lower()}: {len(evidence.intervals)} interval(s) found.",
                timestamps=timestamps_str,
                sensor_modality=modality_str,
                sensor_channels=channels_str,
                explanation=expl_str,
                confidence=evidence.confidence,
                raw_evidence=evidence.to_dict()
            ))

        # 10. General Identification Query
        dominant_act = evidence.result_value
        default_ans = StructuredAnswer(
            answer=f"{dominant_act}",
            activity_event=dominant_act,
            evidence=f"Dominant activity detected from multi-axis inertial features: {evidence.calculation_trace}.",
            timestamps=timestamps_str,
            sensor_modality=modality_str,
            sensor_channels=channels_str,
            explanation=f"Accelerometer variance, gravity orientation, and gyroscope rotation signatures throughout the window correspond with highest probability to {dominant_act}.",
            confidence=evidence.confidence,
            raw_evidence=evidence.to_dict()
        )

        return self._maybe_enrich_with_nim(plan, evidence, default_ans)

    def _maybe_enrich_with_nim(self, plan: QueryPlan, evidence: EvidencePackage, ans: StructuredAnswer) -> StructuredAnswer:
        """Enriches the narrative explanation using NVIDIA NIM while preserving verified numbers."""
        if (
            evidence.operation not in {
                "context_filter", "co_occurrence", "context_presence",
                "location_duration", "location_count", "location_identification",
                "place_transition",
            }
            and self.nim_client
            and self.nim_client.is_available()
            and ans.is_supported
            and evidence.intervals
        ):
            try:
                nim_expl = self.nim_client.synthesize_grounded_explanation(
                    question=plan.raw_question or "User query",
                    answer=ans.answer,
                    activity_event=ans.activity_event,
                    timestamps_str=ans.timestamps,
                    modalities_str=ans.sensor_modality,
                    channels_str=ans.sensor_channels,
                    calculation_trace=evidence.calculation_trace
                )
                if nim_expl:
                    ans.explanation = nim_expl
            except Exception as e:
                print(f"[NIM Explanation Hook] Fallback to standard explanation: {e}")
        return ans
