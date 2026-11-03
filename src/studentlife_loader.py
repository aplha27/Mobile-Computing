"""
studentlife_loader.py
=====================
StudentLife Dataset Integration for SensorQA.

StudentLife (Dartmouth, 2014) provides continuous phone-sensed data from 48 students
over a 10-week term. It includes:
  - GPS traces with lat/lon + travel state (stationary / moving)
  - Wi-Fi-based indoor building location (in[building] / near[building])
  - Android activity recognition (0=stationary, 1=walking, 2=running, 3=unknown)
  - Rich EMA (Ecological Momentary Assessment) self-reports: mood, stress, sleep,
    social, exercise, dining, study spaces, etc.

This module produces:
  - ActivityTimeline  (from sensing/activity codes)
  - ContextTimeline   (from GPS travelstate + EMA social/study/dining labels)
  - LocationTimeline  (from wifi_location building labels + GPS place clustering)

Reference:
  Wang, R. et al. "StudentLife: Assessing Mental Health, Academic Performance and
  Behavioral Trends of College Students using Smartphones". UbiComp 2014.
"""

import zipfile
import csv
import io
import re
import json
import math
from typing import Dict, List, Optional, Tuple, Any

# ── Activity code mapping (Android ActivityRecognition API) ────────────────
STUDENTLIFE_ACTIVITY_MAP = {
    0: "stationary",   # maps to sitting / standing_in_place
    1: "walking",
    2: "running",
    3: None,           # unknown — excluded
}

# Map StudentLife activity codes → SensorQA taxonomy
STUDENTLIFE_TO_SENSORQA = {
    0: "sitting",           # conservative: stationary → sitting
    1: "walking",
    2: "running",
}

# ── Building → semantic place type ────────────────────────────────────────
# Map Dartmouth building names to coarse SensorQA place labels
BUILDING_PLACE_MAP = {
    # Libraries / study
    "baker-berry": "library",
    "dana-library": "library",
    "feldberg_library": "library",
    "kresge": "library",

    # Sports / gym
    "berry_sports_center": "gym",
    "alumni-gym": "gym",
    "sport-venues-press": "gym",
    "mclaughlin": "gym",

    # Dining / restaurants
    "53_commons": "dining_hall",
    "foco": "dining_hall",
    "collis": "dining_hall",
    "hop": "dining_hall",
    "7-lebanon": "restaurant",
    "HanoverInn": "restaurant",

    # Academic buildings
    "dartmouth_hall": "class",
    "silsby": "class",
    "moore": "class",
    "thornton": "class",
    "carpenter": "class",
    "carpenterhall": "class",
    "sudikoff": "class",
    "kemeny": "class",
    "fairchild": "class",
    "cummings": "class",
    "murdough": "class",
    "tuck_hall": "class",
    "Tuck_hall": "class",
    "hallgarten": "class",
    "blunt_alumni_center": "work",

    # Dorms / residential
    "east-wheelock": "dorm",
    "north-park": "dorm",
    "south-park": "dorm",
    "west-wheelock": "dorm",
    "fahey-mclane": "dorm",
    "topliff": "dorm",
    "butterfield": "dorm",
    "gile": "dorm",
    "burke": "dorm",
    "buchanan": "dorm",
    "cutter-north": "dorm",
    "dewey": "dorm",
    "fayerweather": "dorm",
    "byrnehall": "dorm",
    "batrlett": "dorm",
    "currier": "dorm",
    "chasehall": "dorm",
    "wentworth": "dorm",
    "richardson": "dorm",
}

# SensorQA place label equivalences (maps StudentLife → ExtraSensory-style labels)
PLACE_TO_SENSORQA = {
    "library":     "main_workplace",  # study = work analog
    "class":       "main_workplace",
    "work":        "main_workplace",
    "gym":         "gym",
    "dining_hall": "restaurant",
    "restaurant":  "restaurant",
    "dorm":        "home",            # dorm = home analog for students
}

# ── Context label mapping from EMA ────────────────────────────────────────
# EMA types that contribute context labels usable in SensorQA
EMA_CONTEXT_MAP = {
    "Social":    "with_friends",
    "Exercise":  "exercising",
    "Study Spaces": "studying",
    "Sleep":     "sleeping",
    "Dining Halls": "at_dining",
}


class StudentLifeLoader:
    """
    Loads and parses StudentLife dataset from a ZIP archive.
    Produces activity, context, and location data compatible with SensorQA timelines.
    """

    def __init__(self, zip_path: str = "studentlife.zip"):
        self.zip_path = zip_path
        self._zip: Optional[zipfile.ZipFile] = None
        self._names: List[str] = []

    def _open(self) -> zipfile.ZipFile:
        if self._zip is None:
            self._zip = zipfile.ZipFile(self.zip_path, "r")
            self._names = self._zip.namelist()
        return self._zip

    def list_users(self) -> List[str]:
        """Returns list of user IDs available in the archive."""
        z = self._open()
        users = set()
        for name in self._names:
            m = re.search(r"activity_u(\d+)\.csv", name)
            if m:
                users.add(m.group(1))
        return sorted(users)

    # ── GPS ───────────────────────────────────────────────────────────────

    def load_gps(self, user_id: str) -> List[Dict[str, Any]]:
        """
        Loads GPS trace for a user.
        Returns list of dicts with: time, lat, lon, speed, travelstate, accuracy.
        """
        z = self._open()
        path = f"dataset/sensing/gps/gps_u{user_id}.csv"
        if path not in self._names:
            return []

        rows = []
        with z.open(path) as f:
            reader = csv.DictReader(io.TextIOWrapper(f, encoding="utf-8", errors="replace"))
            for row in reader:
                try:
                    rows.append({
                        "time": int(row.get("time", 0)),
                        "lat": float(row.get("latitude", 0)),
                        "lon": float(row.get("longitude", 0)),
                        "speed": float(row.get("speed", 0)),
                        "accuracy": float(row.get("accuracy", 9999)),
                        "travelstate": row.get("travelstate", "").strip(),
                    })
                except (ValueError, KeyError):
                    continue
        return sorted(rows, key=lambda r: r["time"])

    # ── Activity sensing ─────────────────────────────────────────────────

    def load_activity_sensing(self, user_id: str) -> List[Tuple[int, int]]:
        """
        Loads Android ActivityRecognition data.
        Returns list of (unix_timestamp, activity_code) tuples.
        Codes: 0=stationary, 1=walking, 2=running, 3=unknown.
        """
        z = self._open()
        path = f"dataset/sensing/activity/activity_u{user_id}.csv"
        if path not in self._names:
            return []

        rows = []
        with z.open(path) as f:
            for line in io.TextIOWrapper(f, encoding="utf-8", errors="replace"):
                line = line.strip()
                if not line or "timestamp" in line.lower() or "activity" in line.lower():
                    continue
                parts = line.split(",")
                if len(parts) >= 2:
                    try:
                        ts = int(parts[0].strip())
                        code = int(parts[1].strip())
                        rows.append((ts, code))
                    except ValueError:
                        continue
        return sorted(rows, key=lambda x: x[0])

    # ── Wi-Fi location ───────────────────────────────────────────────────

    def load_wifi_location(self, user_id: str) -> List[Dict[str, Any]]:
        """
        Loads Wi-Fi based indoor location labels.
        Returns list of dicts with: time, raw_label, buildings (list), relation (in/near).
        """
        z = self._open()
        path = f"dataset/sensing/wifi_location/wifi_location_u{user_id}.csv"
        if path not in self._names:
            return []

        rows = []
        with z.open(path) as f:
            for line in io.TextIOWrapper(f, encoding="utf-8", errors="replace"):
                line = line.strip()
                if not line or line.startswith("time"):
                    continue
                parts = line.split(",", 1)
                if len(parts) < 2:
                    continue
                try:
                    ts = int(parts[0].strip())
                except ValueError:
                    continue
                raw = parts[1].strip()
                # Parse in[building1] or near[b1; b2; b3]
                m = re.match(r"(in|near)\[([^\]]*)\]", raw)
                if m:
                    relation = m.group(1)
                    buildings = [b.strip() for b in m.group(2).split(";") if b.strip()]
                    rows.append({
                        "time": ts,
                        "raw_label": raw,
                        "relation": relation,
                        "buildings": buildings,
                    })
        return sorted(rows, key=lambda r: r["time"])

    # ── EMA ──────────────────────────────────────────────────────────────

    def load_ema(self, user_id: str, ema_type: str) -> List[Dict[str, Any]]:
        """
        Loads EMA self-report data for a given type and user.
        Returns list of response dicts (each has 'resp_time' and survey fields).
        """
        z = self._open()
        path = f"dataset/EMA/response/{ema_type}/{ema_type}_u{user_id}.json"
        if path not in self._names:
            return []
        with z.open(path) as f:
            try:
                return json.load(f)
            except Exception:
                return []

    # ── Build session ────────────────────────────────────────────────────

    def build_session(self, user_id: str) -> Dict[str, Any]:
        """
        Builds a complete SensorQA-compatible session dict for a StudentLife user.

        Returns:
        {
          session_id: str,
          dataset: "StudentLife",
          user: str,
          activity_intervals: [...],   # for ActivityTimeline
          context_labels: [...],       # for ContextTimeline
          location_labels: [...],      # for LocationTimeline
          gps_trace: [...],
          stats: {...}
        }
        """
        gps = self.load_gps(user_id)
        activity_raw = self.load_activity_sensing(user_id)
        wifi_locs = self.load_wifi_location(user_id)

        if not activity_raw and not gps:
            return {}

        # Determine session time bounds from all sources
        all_times = (
            [r["time"] for r in gps]
            + [t for t, _ in activity_raw]
            + [r["time"] for r in wifi_locs]
        )
        if not all_times:
            return {}

        t_start = min(all_times)
        t_end = max(all_times)
        session_duration = t_end - t_start

        # ── Build activity intervals ─────────────────────────────────────
        # Segment activity_raw into contiguous same-code intervals
        activity_intervals = []
        if activity_raw:
            prev_code, prev_ts = activity_raw[0]
            for ts, code in activity_raw[1:]:
                if code != prev_code or (ts - prev_ts) > 300:  # 5-min gap → new interval
                    sensorqa_act = STUDENTLIFE_TO_SENSORQA.get(prev_code)
                    if sensorqa_act and (prev_ts - t_start) >= 0:
                        activity_intervals.append({
                            "activity": sensorqa_act,
                            "start": float(prev_ts - t_start),
                            "end": float(ts - t_start),
                            "confidence": 0.80 if prev_code in (1, 2) else 0.65,
                            "source": "StudentLife-ActivityRecognition",
                        })
                    prev_code, prev_ts = code, ts
            # Final interval
            sensorqa_act = STUDENTLIFE_TO_SENSORQA.get(prev_code)
            if sensorqa_act:
                activity_intervals.append({
                    "activity": sensorqa_act,
                    "start": float(prev_ts - t_start),
                    "end": float(t_end - t_start),
                    "confidence": 0.80 if prev_code in (1, 2) else 0.65,
                    "source": "StudentLife-ActivityRecognition",
                })

        # ── Build GPS travelstate context labels ─────────────────────────
        # "moving" / "stationary" from GPS → context labels
        context_labels = []
        if gps:
            prev_state = gps[0]["travelstate"]
            prev_ts = gps[0]["time"]
            for row in gps[1:]:
                state = row["travelstate"]
                if state != prev_state or (row["time"] - prev_ts) > 600:
                    if prev_state in ("moving", "stationary"):
                        label = "outdoors_moving" if prev_state == "moving" else "stationary_outdoors"
                        context_labels.append({
                            "label": label,
                            "start": float(prev_ts - t_start),
                            "end": float(row["time"] - t_start),
                            "source": "GPS-travelstate",
                        })
                    prev_state, prev_ts = state, row["time"]

        # ── Build location labels from wifi_location ─────────────────────
        location_labels = []
        if wifi_locs:
            prev_place = None
            prev_ts = wifi_locs[0]["time"]

            for row in wifi_locs:
                buildings = row.get("buildings", [])
                relation = row.get("relation", "near")

                # Determine dominant place label (use first "in" building)
                place_label = None
                for b in buildings:
                    btype = BUILDING_PLACE_MAP.get(b)
                    if btype:
                        sensorqa_place = PLACE_TO_SENSORQA.get(btype, btype)
                        place_label = sensorqa_place
                        if relation == "in":
                            break   # prefer confirmed "in" matches

                if place_label != prev_place:
                    if prev_place and (row["time"] - prev_ts) < 7200:  # max 2-hr interval
                        location_labels.append({
                            "label": prev_place,
                            "start": float(prev_ts - t_start),
                            "end": float(row["time"] - t_start),
                            "source": "StudentLife-WiFiLocation",
                        })
                    prev_place = place_label
                    prev_ts = row["time"]

            # Close last interval
            if prev_place:
                location_labels.append({
                    "label": prev_place,
                    "start": float(prev_ts - t_start),
                    "end": float(t_end - t_start),
                    "source": "StudentLife-WiFiLocation",
                })

        # ── EMA-based context (exercise, social, dining) ──────────────────
        for ema_type, context_label in EMA_CONTEXT_MAP.items():
            ema_data = self.load_ema(user_id, ema_type)
            for entry in ema_data:
                resp_time = entry.get("resp_time", 0)
                if resp_time and t_start <= resp_time <= t_end:
                    # EMA is a point-in-time report → create a ±30-min window
                    rel = float(resp_time - t_start)
                    context_labels.append({
                        "label": context_label,
                        "start": max(0.0, rel - 1800),
                        "end": min(float(session_duration), rel + 1800),
                        "source": f"EMA-{ema_type}",
                    })

        return {
            "session_id": f"studentlife-u{user_id}",
            "dataset": "StudentLife",
            "user": f"u{user_id}",
            "t_start_unix": t_start,
            "t_end_unix": t_end,
            "session_duration_sec": session_duration,
            "activity_intervals": activity_intervals,
            "context_labels": context_labels,
            "location_labels": location_labels,
            "gps_trace": [
                {
                    "rel_time": float(r["time"] - t_start),
                    "lat": r["lat"],
                    "lon": r["lon"],
                    "speed": r["speed"],
                    "travelstate": r["travelstate"],
                }
                for r in gps
            ],
            "stats": {
                "gps_points": len(gps),
                "activity_sensing_points": len(activity_raw),
                "wifi_location_points": len(wifi_locs),
                "activity_intervals_built": len(activity_intervals),
                "context_label_intervals": len(context_labels),
                "location_label_intervals": len(location_labels),
            },
        }

    def close(self):
        if self._zip:
            self._zip.close()
            self._zip = None

