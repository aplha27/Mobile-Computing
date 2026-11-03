"""
nim_llm.py
==========
Optional NVIDIA NIM (Inference Microservice) Client Integration.
Integrates meta/llama-3.2-11b-vision-instruct for:
  1. An available query-planning helper (not used by the current rule-based planner)
  2. Grounded fact-bound activity explanation synthesis
Strictly adheres to the Core Principle: The LLM does NOT do arithmetic calculations,
it reasons over and verbalizes the deterministic evidence packages.
"""

import os
import json
import re
from typing import Dict, List, Any, Optional
from openai import OpenAI

DEFAULT_NIM_BASE_URL = "https://integrate.api.nvidia.com/v1"
DEFAULT_NIM_MODEL = "meta/llama-3.2-11b-vision-instruct"


class NvidiaNimClient:
    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = DEFAULT_NIM_BASE_URL,
        model: str = DEFAULT_NIM_MODEL
    ):
        self.api_key = api_key or os.environ.get("NVIDIA_API_KEY", "")
        self.base_url = base_url
        self.model = model
        self.client: Optional[OpenAI] = None
        self._init_client()

    def _init_client(self):
        if self.api_key:
            try:
                self.client = OpenAI(
                    base_url=self.base_url,
                    api_key=self.api_key
                )
            except Exception as e:
                print(f"[NVIDIA NIM] Warning: Could not initialize OpenAI client: {e}")
                self.client = None

    def is_available(self) -> bool:
        return self.client is not None

    def plan_query_with_nim(self, question: str) -> Optional[Dict[str, Any]]:
        """
        Uses NVIDIA NIM to decompose complex, open-world natural language questions into structured query JSON.
        """
        if not self.is_available():
            return None

        prompt = f"""You are the Query Planner for a wearable sensor activity QA system.
Dataset: ExtraSensory precomputed activity features and sparse context/place labels.
The current classifier does not consume raw 25 Hz streams.
7 Target Activities:
1. lying_down
2. sitting
3. standing_in_place
4. standing_and_moving
5. walking
6. running
7. bicycling

Question: "{question}"

Instructions:
Decompose this question into JSON with the following schema:
{{
  "operation": "duration" | "count" | "onset" | "offset" | "comparison" | "transition" | "verification" | "identification" | "open_world_reasoning" | "unsupported",
  "target_activity": "activity_name" or null,
  "compare_activity": "activity_name" or null,
  "target_activities_group": ["activity1", ...] or [],
  "from_activity": "activity_name" or null,
  "to_activity": "activity_name" or null,
  "time_bounds": {{"t_min": float_or_null, "t_max": float_or_null}},
  "is_supported": true | false,
  "unsupported_reason": string_or_null
}}

Important: If the question asks for audio, food, GPS coordinates, heart rate, or screens, mark is_supported=false and operation="unsupported".
Output ONLY valid JSON:"""

        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a precise JSON-only expert query planner."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=300,
                temperature=0.1
            )
            content = resp.choices[0].message.content.strip()
            # Extract JSON block
            m = re.search(r'\{.*\}', content, re.DOTALL)
            if m:
                return json.loads(m.group(0))
        except Exception as e:
            print(f"[NVIDIA NIM Query Planner] Error: {e}")
        return None

    def synthesize_grounded_explanation(
        self,
        question: str,
        answer: str,
        activity_event: str,
        timestamps_str: str,
        modalities_str: str,
        channels_str: str,
        calculation_trace: str
    ) -> Optional[str]:
        """
        Uses NVIDIA NIM to formulate rich, factual, grounded narrative explanations
        strictly adhering to the deterministic calculation trace and sensor provenance.
        """
        if not self.is_available():
            return None

        prompt = f"""You are the Grounded Explanation Generator for Challenge 1 (Wearable Sensor Activity QA).
User Question: "{question}"

Verified Deterministic Evidence:
- Final Answer: {answer}
- Activity/Event: {activity_event}
- Detected Timestamps: {timestamps_str}
- Sensor Modality: {modalities_str}
- Sensor Channels: {channels_str}
- Deterministic Math Trace: {calculation_trace}

Instructions:
Write a concise, professional 2-3 sentence scientific explanation linking the final answer to the deterministic mathematical trace and the 6-axis inertial sensor kinematics (accelerometer & gyroscope).
DO NOT invent any timestamps, activities, or numbers outside the evidence provided.

Explanation:"""

        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a grounded scientific reasoning system. Always stay strictly true to the provided sensor facts."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=200,
                temperature=0.2
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            print(f"[NVIDIA NIM Explanation] Error: {e}")
        return None
