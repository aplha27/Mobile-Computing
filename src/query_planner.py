"""
query_planner.py
================
Rule-based Question Decomposition and Query Planning Engine (SensorChat + JARVIS paradigm).
Converts natural language user questions into structured, machine-executable query plans.
Handles:
  1. Activity grounding & open-world semantic mapping
  2. Query operation identification (duration, count, onset, offset, compare, transition, identification, open_world_reasoning)
  3. Temporal constraint & scope extraction (time windows, relative bounds)
  4. Truthful failure classification for out-of-scope/unsupported queries
"""

import re
from typing import Dict, List, Any, Optional, Tuple

from src.dataset_loader import TARGET_ACTIVITIES, ACTIVITY_DISPLAY_NAMES
from src.nim_llm import NvidiaNimClient


# Synonym and semantic mapping dictionary for open-world activity reasoning
ACTIVITY_SYNONYMS: Dict[str, List[str]] = {
    "lying_down": [
        "lying", "lying down", "lay down", "laying", "reclining", "resting in bed",
        "in bed", "sleeping position", "horizontal posture", "bed rest"
    ],
    "sitting": [
        "sitting", "sit", "sat", "seated", "chair", "desk work", "couch",
        "sedentary posture", "sitting down"
    ],
    "standing_in_place": [
        "standing in place", "standing still", "stand still", "standing stationary",
        "stood still", "upright stationary", "standing without moving"
    ],
    "standing_and_moving": [
        "standing and moving", "strolling", "fidgeting while standing", "pacing",
        "moving around while standing", "shifting weight", "light standing movement"
    ],
    "walking": [
        "walking", "walk", "walked", "stroll", "foot locomotion", "stepping",
        "pedestrian movement", "on foot", "taking a walk"
    ],
    "running": [
        "running", "run", "ran", "jogging", "jog", "jogged", "sprinting", "sprint",
        "high intensity locomotion", "vigorous running"
    ],
    "bicycling": [
        "bicycling", "cycling", "bicycle", "bike", "biking", "pedaling", "rode a bike",
        "riding bike", "wheeled or pedal-based mode of movement", "pedal-based movement",
        "wheeled transport", "cycle"
    ]
}

# Open-world semantic category mappings
SEMANTIC_CATEGORIES: Dict[str, Dict[str, Any]] = {
    "wheeled_pedal_transport": {
        "patterns": ["wheeled", "pedal", "bike", "cycling", "bicycle"],
        "target_activities": ["bicycling"],
        "semantic_name": "Wheeled or pedal-based movement"
    },
    "sedentary_behavior": {
        "patterns": ["sedentary", "resting", "inactive", "non-active", "still"],
        "target_activities": ["sitting", "lying_down"],
        "semantic_name": "Sedentary / Rest postures"
    },
    "locomotion": {
        "patterns": ["locomotion", "traveling on foot", "moving from place to place", "ambulating"],
        "target_activities": ["walking", "running"],
        "semantic_name": "Foot locomotion"
    },
    "vigorous_exercise": {
        "patterns": ["vigorous", "high-intensity", "cardio", "intense exercise", "heavy workout"],
        "target_activities": ["running", "bicycling"],
        "semantic_name": "Vigorous aerobic activity"
    },
    "upright_posture": {
        "patterns": ["upright", "on their feet", "standing up"],
        "target_activities": ["standing_in_place", "standing_and_moving", "walking", "running"],
        "semantic_name": "Upright postures and movement"
    }
}

# Out-of-scope topics for truthful rejection (DrHouse / Challenge 1 principle)
UNSUPPORTED_DOMAINS = [
    "eating", "food", "dinner", "lunch", "breakfast", "meal",
    "drink", "drinking", "alcohol", "water",
    "heart rate", "ecg", "blood pressure", "spo2", "temperature",
    "talking", "speech", "audio", "conversation", "who was speaking",
    "gps coordinates", "raw gps", "gps", "exact coordinates", "exact location",
    "address", "city", "building name", "latitude", "longitude",
    "how far", "distance", "route", "speed",
    "screen", "app", "phone call", "text message", "whatsapp"
]

CONTEXT_SYNONYMS: Dict[str, List[str]] = {
    "indoors": ["indoors", "indoor", "inside"],
    "outdoors": ["outdoors", "outdoor", "outside"],
    "in_a_car": ["in a car", "inside a car", "in the car", "car"],
    "on_a_bus": ["on a bus", "in a bus", "bus"],
    "driving": ["driving", "as the driver"],
    "passenger": ["as a passenger", "riding as a passenger"],
    "with_friends": ["with friends", "with my friends"],
    "with_coworkers": ["with coworkers", "with co-workers", "with colleagues"],
    "in_a_meeting": ["in a meeting", "during a meeting"],
    "in_class": ["in class", "during class"],
}

LOCATION_SYNONYMS: Dict[str, List[str]] = {
    "home": ["at home", "home"],
    "work": ["at work", "workplace", "office", "work"],
    "restaurant": ["at a restaurant", "at the restaurant", "restaurant"],
    "gym": ["at the gym", "at gym", "gym"],
    "beach": ["at the beach", "beach"],
    "school": ["at school", "school"],
}


class QueryPlan:
    def __init__(
        self,
        operation: str,
        target_activity: Optional[str] = None,
        compare_activity: Optional[str] = None,
        target_activities_group: Optional[List[str]] = None,
        t_min: Optional[float] = None,
        t_max: Optional[float] = None,
        target_timestamp: Optional[float] = None,
        is_supported: bool = True,
        unsupported_reason: Optional[str] = None,
        semantic_category: Optional[str] = None,
        from_activity: Optional[str] = None,
        to_activity: Optional[str] = None,
        context_label: Optional[str] = None,
        location_label: Optional[str] = None,
        from_location: Optional[str] = None,
        to_location: Optional[str] = None,
        query_metric: str = "duration",
        raw_question: str = ""
    ):
        self.operation = operation
        self.target_activity = target_activity
        self.compare_activity = compare_activity
        self.target_activities_group = target_activities_group or []
        self.t_min = t_min
        self.t_max = t_max
        self.target_timestamp = target_timestamp
        self.is_supported = is_supported
        self.unsupported_reason = unsupported_reason
        self.semantic_category = semantic_category
        self.from_activity = from_activity
        self.to_activity = to_activity
        self.context_label = context_label
        self.location_label = location_label
        self.from_location = from_location
        self.to_location = to_location
        self.query_metric = query_metric
        self.raw_question = raw_question

    def to_dict(self) -> Dict[str, Any]:
        return {
            "operation": self.operation,
            "target_activity": self.target_activity,
            "compare_activity": self.compare_activity,
            "target_activities_group": self.target_activities_group,
            "time_range": {
                "t_min": self.t_min,
                "t_max": self.t_max
            } if (self.t_min is not None or self.t_max is not None) else None,
            "target_timestamp": self.target_timestamp,
            "is_supported": self.is_supported,
            "unsupported_reason": self.unsupported_reason,
            "semantic_category": self.semantic_category,
            "from_activity": self.from_activity,
            "to_activity": self.to_activity,
            "context_label": self.context_label,
            "location_label": self.location_label,
            "from_location": self.from_location,
            "to_location": self.to_location,
            "query_metric": self.query_metric,
            "raw_question": self.raw_question
        }


class QueryPlanner:
    """
    Expert Question Decomposition Engine.
    Parses natural language into formal structured query plans using deterministic rules.
    """
    def __init__(self, nim_client: Optional[NvidiaNimClient] = None, use_nim: bool = True):
        self.nim_client = nim_client or (NvidiaNimClient() if use_nim else None)

    def _match_activity(self, text: str) -> Optional[str]:
        """Matches textual mention to one of the 7 standard activity keys."""
        text_lower = text.lower()
        # Direct canonical name match
        for act in TARGET_ACTIVITIES:
            if act in text_lower:
                return act
            if act.replace("_", " ") in text_lower:
                return act

        # Match synonyms (longest match first)
        all_matches = []
        for act, syn_list in ACTIVITY_SYNONYMS.items():
            for syn in syn_list:
                # Word boundary match
                pattern = r'\b' + re.escape(syn) + r'\b'
                if re.search(pattern, text_lower):
                    all_matches.append((len(syn), act))

        if all_matches:
            all_matches.sort(reverse=True, key=lambda x: x[0])
            return all_matches[0][1]

        return None

    @staticmethod
    def _match_label(text: str, aliases: Dict[str, List[str]]) -> Optional[str]:
        matches = []
        for label, synonyms in aliases.items():
            for synonym in synonyms:
                if re.search(r"\b" + re.escape(synonym) + r"\b", text):
                    matches.append((len(synonym), label))
        return max(matches)[1] if matches else None

    def _extract_time_bounds(self, text: str) -> Tuple[Optional[float], Optional[float], Optional[float]]:
        """
        Extracts temporal constraints such as:
        - "between 100 and 500 seconds" / "between 10:00 and 20:00"
        - "at second 300" / "at timestamp 1500"
        - "in the first 30 minutes"
        """
        text_lower = text.lower()
        t_min, t_max, target_ts = None, None, None

        # Pattern: between X and Y seconds / s / mins
        m_between = re.search(r'between\s+(\d+(?:\.\d+)?)\s*(?:and|to|-)\s*(\d+(?:\.\d+)?)\s*(s|sec|seconds|min|mins|minutes)?', text_lower)
        if m_between:
            v1 = float(m_between.group(1))
            v2 = float(m_between.group(2))
            unit = m_between.group(3) or "s"
            multiplier = 60.0 if "min" in unit else 1.0
            t_min, t_max = min(v1, v2) * multiplier, max(v1, v2) * multiplier
            return t_min, t_max, target_ts

        # Pattern: from X to Y s/seconds/minutes
        m_from_to = re.search(r'from\s+(\d+(?:\.\d+)?)\s*(?:to|-)\s*(\d+(?:\.\d+)?)\s*(s|sec|seconds|min|mins|minutes)?', text_lower)
        if m_from_to:
            v1 = float(m_from_to.group(1))
            v2 = float(m_from_to.group(2))
            unit = m_from_to.group(3) or "s"
            multiplier = 60.0 if "min" in unit else 1.0
            t_min, t_max = min(v1, v2) * multiplier, max(v1, v2) * multiplier
            return t_min, t_max, target_ts

        # Pattern: first X seconds / minutes
        m_first = re.search(r'(?:in\s+the\s+)?first\s+(\d+(?:\.\d+)?)\s*(s|sec|seconds|min|mins|minutes)', text_lower)
        if m_first:
            v = float(m_first.group(1))
            unit = m_first.group(2)
            multiplier = 60.0 if "min" in unit else 1.0
            t_min, t_max = 0.0, v * multiplier
            return t_min, t_max, target_ts

        # Pattern: at timestamp / second X
        m_at = re.search(r'(?:at|around)\s+(?:timestamp|time|second|sec)\s+(\d+(?:\.\d+)?)', text_lower)
        if m_at:
            target_ts = float(m_at.group(1))
            return t_min, t_max, target_ts

        return t_min, t_max, target_ts

    def plan_query(self, question: str) -> QueryPlan:
        """
        Decomposes natural language question into an executable QueryPlan.
        """
        q_lower = question.strip().lower()
        t_min, t_max, target_ts = self._extract_time_bounds(q_lower)

        # 1. Check for unsupported domain questions (DrHouse truthful rejection)
        for term in UNSUPPORTED_DOMAINS:
            if re.search(r'\b' + re.escape(term) + r'\b', q_lower):
                if term == "gps" and self._match_label(q_lower, LOCATION_SYNONYMS):
                    continue
                return QueryPlan(
                    operation="unsupported",
                    is_supported=False,
                    unsupported_reason=(
                        f"The question inquires about '{term}', which is not available "
                        "in the provided ExtraSensory labels or sensor fields."
                    ),
                    raw_question=question
                )

        # Place-to-place transitions are based only on annotated place labels.
        place_transition = re.search(
            r"\bfrom\s+([a-z\s_-]+?)\s+(?:to|into)\s+([a-z\s_-]+?)(?:[?.!,]|$)",
            q_lower,
        )
        if place_transition:
            from_location = self._match_label(place_transition.group(1), LOCATION_SYNONYMS)
            to_location = self._match_label(place_transition.group(2), LOCATION_SYNONYMS)
            if from_location and to_location:
                return QueryPlan(
                    operation="place_transition",
                    from_location=from_location,
                    to_location=to_location,
                    t_min=t_min,
                    t_max=t_max,
                    raw_question=question,
                )

        context_label = self._match_label(q_lower, CONTEXT_SYNONYMS)
        location_label = self._match_label(q_lower, LOCATION_SYNONYMS)
        primary_activity = self._match_activity(q_lower)
        asks_for_count = any(
            word in q_lower
            for word in ["how many times", "count", "number of times", "frequency", "how often", "visits"]
        )
        asks_for_duration = any(
            word in q_lower
            for word in [
                "how long", "duration", "total time", "amount of time",
                "how much time", "how many minutes", "how many seconds",
            ]
        )
        asks_yes_no = q_lower.startswith(("did ", "was ", "were ", "is ", "are ", "have "))

        if primary_activity and (context_label or location_label):
            operation = "co_occurrence" if asks_yes_no and not asks_for_duration else "context_filter"
            return QueryPlan(
                operation=operation,
                target_activity=primary_activity,
                context_label=context_label,
                location_label=location_label,
                query_metric="count" if asks_for_count else "duration",
                t_min=t_min,
                t_max=t_max,
                raw_question=question,
            )

        if location_label:
            if asks_yes_no:
                operation = "context_presence"
            elif asks_for_count:
                operation = "location_count"
            elif asks_for_duration:
                operation = "location_duration"
            else:
                operation = "location_identification"
            return QueryPlan(
                operation=operation,
                location_label=location_label,
                query_metric="count" if asks_for_count else "duration",
                target_timestamp=target_ts,
                t_min=t_min,
                t_max=t_max,
                raw_question=question,
            )

        if context_label:
            operation = "context_presence" if asks_yes_no else "context_filter"
            return QueryPlan(
                operation=operation,
                context_label=context_label,
                query_metric="count" if asks_for_count else "duration",
                t_min=t_min,
                t_max=t_max,
                raw_question=question,
            )

        if any(
            phrase in q_lower
            for phrase in [
                "where was", "where did i go", "what location", "which place",
                "what places", "places did i visit", "where am", "my location",
            ]
        ):
            return QueryPlan(
                operation="location_identification",
                target_timestamp=target_ts,
                t_min=t_min,
                t_max=t_max,
                raw_question=question,
            )

        # 2. Check for Transition queries ("transition from X to Y", "switch from sitting to walking")
        m_trans = re.search(r'(?:transition|switch|change|shift)(?:ed|ing)?\s+(?:from\s+)?([a-z\s_-]+?)\s+(?:to|into)\s+([a-z\s_-]+)', q_lower)
        if m_trans:
            act1 = self._match_activity(m_trans.group(1))
            act2 = self._match_activity(m_trans.group(2))
            if act1 and act2:
                return QueryPlan(
                    operation="transition",
                    from_activity=act1,
                    to_activity=act2,
                    t_min=t_min,
                    t_max=t_max,
                    raw_question=question
                )

        # 3. Check for Comparative queries ("more time walking or sitting", "compare walking and running")
        if any(w in q_lower for w in ["more time", "less time", "longer", "shorter", "compare", "versus", " vs "]):
            # Find two activities
            matched_acts = []
            for act, syn_list in ACTIVITY_SYNONYMS.items():
                for syn in syn_list:
                    if re.search(r'\b' + re.escape(syn) + r'\b', q_lower):
                        if act not in matched_acts:
                            matched_acts.append(act)
                        break
            if len(matched_acts) >= 2:
                return QueryPlan(
                    operation="comparison",
                    target_activity=matched_acts[0],
                    compare_activity=matched_acts[1],
                    t_min=t_min,
                    t_max=t_max,
                    raw_question=question
                )

        # 4. Check for Open-World Semantic Category queries (e.g. "wheeled or pedal-based mode of movement")
        for cat_key, cat_meta in SEMANTIC_CATEGORIES.items():
            if any(p in q_lower for p in cat_meta["patterns"]):
                # If question explicitly mentions open world reasoning or category
                if any(w in q_lower for w in ["wheeled", "pedal", "sedentary", "locomotion", "vigorous", "upright"]):
                    return QueryPlan(
                        operation="open_world_reasoning",
                        target_activities_group=cat_meta["target_activities"],
                        semantic_category=cat_meta["semantic_name"],
                        t_min=t_min,
                        t_max=t_max,
                        raw_question=question
                    )

        # Match single primary activity
        primary_activity = self._match_activity(q_lower)

        # 5. Check for Temporal Onset / Start queries ("when did running begin", "start time of walking")
        if any(w in q_lower for w in ["when did", "when does", "start time", "begin", "onset", "started"]):
            if primary_activity:
                return QueryPlan(
                    operation="onset",
                    target_activity=primary_activity,
                    t_min=t_min,
                    t_max=t_max,
                    raw_question=question
                )

        # 6. Check for Temporal Offset / End queries ("when did walking end", "finish time")
        if any(w in q_lower for w in ["when ended", "end time", "finish", "stopped", "conclude"]):
            if primary_activity:
                return QueryPlan(
                    operation="offset",
                    target_activity=primary_activity,
                    t_min=t_min,
                    t_max=t_max,
                    raw_question=question
                )

        # 7. Check for Duration queries ("how long did I walk", "total duration", "how many minutes/seconds")
        if any(w in q_lower for w in ["how long", "duration", "total time", "amount of time", "how much time", "how many minutes", "how many seconds"]):
            if primary_activity:
                return QueryPlan(
                    operation="duration",
                    target_activity=primary_activity,
                    t_min=t_min,
                    t_max=t_max,
                    raw_question=question
                )

        # 8. Check for Count / Frequency queries ("how many times did I run", "number of occurrences")
        if any(w in q_lower for w in ["how many times", "count", "number of times", "frequency", "how often", "number of sessions"]):
            if primary_activity:
                return QueryPlan(
                    operation="count",
                    target_activity=primary_activity,
                    t_min=t_min,
                    t_max=t_max,
                    raw_question=question
                )

        # 9. Check for Specific Timestamp / Interval Identification queries ("what activity at 500s", "what was the user doing")
        if any(w in q_lower for w in ["what activity", "what was", "which activity", "is the user", "was the user"]):
            if primary_activity:
                # Binary verification ("Is the user running?")
                return QueryPlan(
                    operation="verification",
                    target_activity=primary_activity,
                    target_timestamp=target_ts,
                    t_min=t_min,
                    t_max=t_max,
                    raw_question=question
                )
            else:
                # General identification ("What was the user doing?")
                return QueryPlan(
                    operation="identification",
                    target_timestamp=target_ts,
                    t_min=t_min,
                    t_max=t_max,
                    raw_question=question
                )

        # 10. Check for Interval listing query ("give me the walking intervals", "timestamps of running")
        if any(w in q_lower for w in ["intervals", "timestamps", "when was", "show periods"]):
            if primary_activity:
                return QueryPlan(
                    operation="intervals",
                    target_activity=primary_activity,
                    t_min=t_min,
                    t_max=t_max,
                    raw_question=question
                )

        # Fallback query plan
        if primary_activity:
            return QueryPlan(
                operation="duration",
                target_activity=primary_activity,
                t_min=t_min,
                t_max=t_max,
                raw_question=question
            )

        return QueryPlan(
            operation="identification",
            t_min=t_min,
            t_max=t_max,
            raw_question=question
        )
