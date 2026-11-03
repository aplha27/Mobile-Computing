"""
web_app.py
==========
Interactive Web Application for Challenge 1: Grounded Activity QA over Wearable Sensors.
FastAPI backend providing REST endpoints for:
  - Session timeline indexing and retrieval
  - 25 Hz 6-channel sensor signal synthesis and waveform streaming
  - Natural Language question decomposition and structured QA reasoning
  - Live multi-task benchmark dashboard and edge profiling metrics
"""

import os
import sys
import json
import pickle
from typing import Dict, List, Any, Optional
import numpy as np
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

# Add repo root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.dataset_loader import ExtraSensoryLoader, TARGET_ACTIVITIES, ACTIVITY_DISPLAY_NAMES
from src.sensor_model import SensorUnderstandingPipeline
from src.context_conditioned_model import ContextConditionedActivityModel
from src.raw_recording_store import RawRecordingStore
from src.pamap2_model import (
    PAMAP2ActivityModel,
    PAMAP2Loader,
    PAMAP2_LABEL_NAMES,
    PAMAP2_ARCHIVE,
    MAX_SUPPORTED_SAMPLE_RATE_HZ,
    MIN_SUPPORTED_SAMPLE_RATE_HZ,
    WINDOW_SEC as PAMAP2_WINDOW_SEC,
    TIMELINE_STEP_SEC as PAMAP2_TIMELINE_STEP_SEC,
)
from src.shl_model import (
    SHL_ACTIVITY_MAP,
    SHL_ARCHIVES,
    SHL_LABEL_NAMES,
    SHL_TIMELINE_STEP_SEC,
    SHLActivityModel,
    SHLLoader,
)
from src.studentlife_loader import StudentLifeLoader
from src.timeline_store import ActivityTimeline, ActivityInterval
from src.context_store import ContextTimeline, LocationTimeline, LabelInterval
from src.query_planner import QueryPlanner
from src.deterministic_engine import DeterministicEngine
from src.answer_generator import AnswerGenerator
from src.evaluator import BenchmarkEvaluator

app = FastAPI(title="Grounded Activity QA", version="1.0.0")

# Setup templates and static
os.makedirs("src/templates", exist_ok=True)
os.makedirs("src/static", exist_ok=True)
templates = Jinja2Templates(directory="src/templates")
app.mount("/static", StaticFiles(directory="src/static"), name="static")

# Global instances
LOADER: Optional[ExtraSensoryLoader] = None
SPLITS: Optional[Dict[str, Any]] = None
SENSOR_MODEL: Optional[SensorUnderstandingPipeline] = None
PLANNER = QueryPlanner()
ENGINE = DeterministicEngine()
GENERATOR = AnswerGenerator()
SESSIONS_CACHE: Dict[str, Dict[str, Any]] = {}
TIMELINES_CACHE: Dict[str, ActivityTimeline] = {}
CONTEXT_TIMELINES_CACHE: Dict[str, ContextTimeline] = {}
LOCATION_TIMELINES_CACHE: Dict[str, LocationTimeline] = {}
BENCHMARK_CACHE: Optional[Dict[str, Any]] = None
CONTEXT_CONDITIONED_MODEL: Optional[ContextConditionedActivityModel] = None
RAW_RECORDINGS = RawRecordingStore()
MAX_RAW_CSV_BYTES = 10 * 1024 * 1024
PAMAP2_MODEL: Optional[PAMAP2ActivityModel] = None
PAMAP2_LOADER: Optional[PAMAP2Loader] = None
PAMAP2_SESSIONS: Dict[str, Dict[str, Any]] = {}
SHL_MODEL: Optional[SHLActivityModel] = None
SHL_LOADER: Optional[SHLLoader] = None
SHL_SESSIONS: Dict[str, Dict[str, Any]] = {}
STUDENTLIFE_LOADER: Optional[StudentLifeLoader] = None
STUDENTLIFE_SESSIONS: Dict[str, Dict[str, Any]] = {}


def initialize_system():
    global LOADER, SPLITS, SENSOR_MODEL, BENCHMARK_CACHE, CONTEXT_CONDITIONED_MODEL
    global PAMAP2_MODEL, PAMAP2_LOADER
    global SHL_MODEL, SHL_LOADER
    global STUDENTLIFE_LOADER
    print("Initializing Grounded Activity QA Web System...")
    LOADER = ExtraSensoryLoader(zip_path="ExtraSensory.per_uuid_features_labels.zip")
    
    # Load dataset splits (loads official CV folds with unseen test users)
    SPLITS = LOADER.load_dataset_split(num_train_users=40, num_val_users=8, num_test_users=12)
    
    SENSOR_MODEL = SensorUnderstandingPipeline()
    model_path = "checkpoints/sensor_model.pkl"
    if os.path.exists(model_path):
        try:
            print(f"Loading sensor model checkpoint: {model_path}")
            SENSOR_MODEL.load(model_path)
            if SENSOR_MODEL.feature_scaler is None:
                print("Checkpoint has no saved feature scaler; retraining pipeline...")
                SENSOR_MODEL.fit_full_pipeline(SPLITS)
                SENSOR_MODEL.save(model_path)
        except Exception as e:
            print(f"Failed to load checkpoint ({e}), retraining pipeline...")
            SENSOR_MODEL.fit_full_pipeline(SPLITS)
            SENSOR_MODEL.save(model_path)
    else:
        print("Training sensor understanding pipeline...")
        SENSOR_MODEL.fit_full_pipeline(SPLITS)
        SENSOR_MODEL.save(model_path)

    context_model_path = "checkpoints/context_conditioned_model.pkl"
    if os.path.exists(context_model_path):
        print(f"Loading context-conditioned model: {context_model_path}")
        CONTEXT_CONDITIONED_MODEL = ContextConditionedActivityModel.load(
            context_model_path
        )
        if CONTEXT_CONDITIONED_MODEL.feature_count != SENSOR_MODEL.input_dim:
            raise ValueError(
                "Context-conditioned checkpoint feature width does not match "
                "the loaded ExtraSensory model."
            )

    pamap2_model_path = "checkpoints/pamap2_activity_model.pkl"
    if os.path.exists(pamap2_model_path) and os.path.exists(PAMAP2_ARCHIVE):
        print(f"Loading PAMAP2 raw-signal model: {pamap2_model_path}")
        PAMAP2_MODEL = PAMAP2ActivityModel.load(pamap2_model_path)
        PAMAP2_LOADER = PAMAP2Loader(PAMAP2_ARCHIVE)
        for session in PAMAP2_LOADER.load_protocol_subjects():
            if not len(session["labels"]):
                continue
            predictions, confidences = PAMAP2_MODEL.predict(session["features"])
            timeline = ActivityTimeline.from_predictions(
                session["timestamps"],
                predictions,
                confidences,
                session_id=session["session_id"],
                window_sec=PAMAP2_TIMELINE_STEP_SEC,
                merge_gap_sec=PAMAP2_TIMELINE_STEP_SEC,
                time_origin=session["recording_start"],
                session_duration=session["recording_duration_sec"],
            )
            timestamps = session["timestamps"]
            origin = session["recording_start"]
            TIMELINES_CACHE[session["session_id"]] = timeline
            CONTEXT_TIMELINES_CACHE[session["session_id"]] = ContextTimeline.from_labels(
                timestamps,
                {},
                session_id=session["session_id"],
                time_origin=origin,
                session_end=timeline.session_end,
            )
            LOCATION_TIMELINES_CACHE[session["session_id"]] = LocationTimeline.from_labels(
                timestamps,
                {},
                session_id=session["session_id"],
                time_origin=origin,
                session_end=timeline.session_end,
            )
            PAMAP2_SESSIONS[session["session_id"]] = session

    shl_model_path = "checkpoints/shl_activity_model.pkl"
    shl_predictions_path = "results/shl_session_predictions.pkl"
    if (
        os.path.exists(shl_model_path)
        and all(os.path.exists(path) for path in SHL_ARCHIVES)
        and not os.path.exists(shl_predictions_path)
    ):
        raise FileNotFoundError(
            "SHL session predictions are missing. Rebuild them with "
            "python -m src.train_shl_model."
        )
    if (
        os.path.exists(shl_model_path)
        and os.path.exists(shl_predictions_path)
        and all(os.path.exists(path) for path in SHL_ARCHIVES)
    ):
        print(f"Loading SHL Hand-IMU model: {shl_model_path}")
        SHL_MODEL = SHLActivityModel.load(shl_model_path)
        SHL_LOADER = SHLLoader()
        with open(shl_predictions_path, "rb") as file:
            prediction_cache = pickle.load(file)
        if (
            prediction_cache.get("version") != 1
            or prediction_cache.get("activity_map") != SHL_ACTIVITY_MAP
        ):
            raise ValueError("SHL session prediction cache is incompatible.")
        archive_sessions = SHL_LOADER.available_sessions()
        cached_sessions = prediction_cache.get("sessions", {})
        expected_session_ids = {
            session["session_id"] for session in archive_sessions
        }
        if set(cached_sessions) != expected_session_ids:
            raise ValueError(
                "SHL session prediction cache does not match the available preview archives. "
                "Rebuild it with python -m src.train_shl_model."
            )
        for session in archive_sessions:
            cached = cached_sessions[session["session_id"]]
            SHL_SESSIONS[session["session_id"]] = session
            timestamps = np.asarray(cached["timestamps"])
            timeline = ActivityTimeline.from_predictions(
                timestamps,
                np.asarray(cached["predictions"]),
                np.asarray(cached["confidences"]),
                session_id=session["session_id"],
                window_sec=SHL_TIMELINE_STEP_SEC,
                merge_gap_sec=SHL_TIMELINE_STEP_SEC,
                time_origin=cached["recording_start"],
                session_duration=cached["recording_duration_sec"],
            )
            TIMELINES_CACHE[session["session_id"]] = timeline
            CONTEXT_TIMELINES_CACHE[session["session_id"]] = ContextTimeline.from_labels(
                timestamps,
                {},
                session_id=session["session_id"],
                time_origin=cached["recording_start"],
                session_end=timeline.session_end,
            )
            LOCATION_TIMELINES_CACHE[session["session_id"]] = LocationTimeline.from_labels(
                timestamps,
                {},
                session_id=session["session_id"],
                time_origin=cached["recording_start"],
                session_end=timeline.session_end,
            )

    # Pre-cache test session timelines
    for session in SPLITS["test_sessions"]:
        uuid = session["uuid"]
        valid = session["valid_mask"]
        ts = session["timestamps"][valid]
        if CONTEXT_CONDITIONED_MODEL is not None:
            preds, confs = CONTEXT_CONDITIONED_MODEL.predict(
                session["raw_features"][valid],
                {
                    label: values[valid]
                    for label, values in session["context_labels"].items()
                },
                {
                    label: values[valid]
                    for label, values in session["location_labels"].items()
                },
            )
        else:
            preds, confs = SENSOR_MODEL.predict_raw_features(
                session["raw_features"][valid]
            )
        timeline = ActivityTimeline.from_predictions(ts, preds, confs, session_id=uuid)
        SESSIONS_CACHE[uuid] = session
        TIMELINES_CACHE[uuid] = timeline
        CONTEXT_TIMELINES_CACHE[uuid] = ContextTimeline.from_labels(
            session["timestamps"],
            session["context_labels"],
            session_id=uuid,
            time_origin=float(ts[0]) if len(ts) else None,
            session_end=timeline.session_end,
        )
        LOCATION_TIMELINES_CACHE[uuid] = LocationTimeline.from_labels(
            session["timestamps"],
            session["location_labels"],
            session_id=uuid,
            time_origin=float(ts[0]) if len(ts) else None,
            session_end=timeline.session_end,
        )

    # ── StudentLife dataset integration ──────────────────────────────────────
    studentlife_zip = "studentlife.zip"
    if os.path.exists(studentlife_zip):
        print(f"Loading StudentLife dataset: {studentlife_zip}")
        STUDENTLIFE_LOADER = StudentLifeLoader(studentlife_zip)
        sl_users = STUDENTLIFE_LOADER.list_users()
        print(f"  Found {len(sl_users)} StudentLife users")
        for user_id in sl_users:
            try:
                session = STUDENTLIFE_LOADER.build_session(user_id)
                if not session or not session.get("activity_intervals"):
                    continue
                sid = session["session_id"]
                STUDENTLIFE_SESSIONS[sid] = session
                dur = session.get("session_duration_sec", 0)
                act_intervals = [
                    ActivityInterval(
                        activity=iv["activity"],
                        start_time=iv["start"],
                        end_time=iv["end"],
                        confidence=iv["confidence"],
                        modality=["accelerometer"],
                        channels=["Acc X", "Acc Y", "Acc Z"],
                    )
                    for iv in session["activity_intervals"]
                ]
                tl = ActivityTimeline(session_id=sid)
                tl.intervals = act_intervals
                tl.session_start = 0.0
                tl.session_end = float(dur)
                tl.total_duration = float(dur)
                TIMELINES_CACHE[sid] = tl

                ctx_tl = ContextTimeline(session_id=sid)
                ctx_tl.session_end = float(dur)
                for ci in session.get("context_labels", []):
                    label = ci["label"]
                    if label not in ctx_tl.intervals:
                        ctx_tl.intervals[label] = []
                        ctx_tl.observed_intervals[label] = []
                    interval_obj = LabelInterval(label=label, start_time=ci["start"], end_time=ci["end"])
                    ctx_tl.intervals[label].append(interval_obj)
                    ctx_tl.observed_intervals[label].append(interval_obj)
                CONTEXT_TIMELINES_CACHE[sid] = ctx_tl

                loc_tl = LocationTimeline(session_id=sid)
                loc_tl.session_end = float(dur)
                for li in session.get("location_labels", []):
                    label = li["label"]
                    if label not in loc_tl.intervals:
                        loc_tl.intervals[label] = []
                        loc_tl.observed_intervals[label] = []
                    interval_obj = LabelInterval(label=label, start_time=li["start"], end_time=li["end"])
                    loc_tl.intervals[label].append(interval_obj)
                    loc_tl.observed_intervals[label].append(interval_obj)
                LOCATION_TIMELINES_CACHE[sid] = loc_tl
            except Exception as e:
                print(f"  [StudentLife] Skipping user {user_id}: {e}")
        print(f"  StudentLife sessions loaded: {len(STUDENTLIFE_SESSIONS)}")

    # Load or generate benchmark results
    results_path = "results/benchmark_results.json"
    if os.path.exists(results_path):
        with open(results_path, "r") as f:
            BENCHMARK_CACHE = json.load(f)
    else:
        evaluator = BenchmarkEvaluator(SENSOR_MODEL, PLANNER, ENGINE, GENERATOR)
        BENCHMARK_CACHE = evaluator.run_full_benchmark(SPLITS)
        os.makedirs("results", exist_ok=True)
        with open(results_path, "w") as f:
            json.dump(BENCHMARK_CACHE, f, indent=2)

    print("System initialization complete!")


class QuestionRequest(BaseModel):
    session_id: str
    question: str


@app.on_event("startup")
def on_startup():
    initialize_system()


@app.get("/", response_class=HTMLResponse)
def get_index(request: Request):
    sessions = list(TIMELINES_CACHE.keys())
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "sessions": sessions,
            "target_activities": TARGET_ACTIVITIES,
            "activity_names": ACTIVITY_DISPLAY_NAMES
        }
    )


@app.get("/api/sessions")
def list_sessions():
    """Returns list of available sessions with summary statistics."""
    res = []
    for uuid, tl in TIMELINES_CACHE.items():
        res.append({
            "session_id": uuid,
            "dataset": (
                "PAMAP2" if uuid in PAMAP2_SESSIONS
                else "SHL" if uuid in SHL_SESSIONS
                else "StudentLife" if uuid in STUDENTLIFE_SESSIONS
                else "ExtraSensory"
            ),
            "display_name": (
                f"PAMAP2 {PAMAP2_SESSIONS[uuid]['subject']}"
                if uuid in PAMAP2_SESSIONS
                else (
                    f"SHL {SHL_SESSIONS[uuid]['subject']} {SHL_SESSIONS[uuid]['date']}"
                    if uuid in SHL_SESSIONS
                    else (
                        f"StudentLife {STUDENTLIFE_SESSIONS[uuid]['user']} ({int(tl.total_duration/3600)}h)"
                        if uuid in STUDENTLIFE_SESSIONS
                        else f"ExtraSensory Session ({uuid[:8]}...)"
                    )
                )
            ),
            "total_duration_sec": tl.total_duration,
            "total_intervals": len(tl.intervals),
            "activities_present": list(set(i.activity for i in tl.intervals)),
            "has_context": uuid in CONTEXT_TIMELINES_CACHE and bool(
                CONTEXT_TIMELINES_CACHE[uuid].label_intervals
            ),
            "has_location": uuid in LOCATION_TIMELINES_CACHE and bool(
                LOCATION_TIMELINES_CACHE[uuid].label_intervals
            ),
        })
    return {"sessions": res}


@app.get("/api/status")
def get_system_status():
    """Reports feature availability without exposing credentials."""
    nim_client = GENERATOR.nim_client
    return {
        "nim_configured": bool(nim_client and nim_client.is_available()),
        "nim_role": "optional activity-explanation writing only",
        "activity_input": (
            "ExtraSensory features, PAMAP2 chest IMU, and SHL Hand IMU"
            if SHL_MODEL is not None
            else (
                "ExtraSensory precomputed features and PAMAP2 raw chest-IMU windows"
                if PAMAP2_MODEL is not None
                else "ExtraSensory precomputed feature rows"
            )
        ),
        "context_conditioned_model": CONTEXT_CONDITIONED_MODEL is not None,
        "activity_model": (
            "Context-conditioned (ExtraSensory annotations)"
            if CONTEXT_CONDITIONED_MODEL is not None
            else "Activity-only baseline"
        ),
        "waveform_source": (
            "measured SHL Hand IMU and PAMAP2 chest IMU; synthetic illustration for ExtraSensory"
            if SHL_MODEL is not None
            else (
                "measured PAMAP2 signals for PAMAP2 sessions; synthetic illustration for ExtraSensory"
                if PAMAP2_MODEL is not None
                else "synthetic illustration"
            )
        ),
        "raw_recording_import": (
            "PAMAP2 model inference for matching 100 Hz recordings"
            if PAMAP2_MODEL is not None
            else "raw six-axis CSV preview only"
        ),
        "pamap2_model_loaded": PAMAP2_MODEL is not None,
        "pamap2_protocol_sessions": len(PAMAP2_SESSIONS),
        "shl_model_loaded": SHL_MODEL is not None,
        "shl_hand_sessions": len(SHL_SESSIONS),
    }


@app.get("/api/pamap2/metrics")
def get_pamap2_metrics():
    metrics_path = "results/pamap2_metrics.json"
    if not os.path.exists(metrics_path):
        raise HTTPException(status_code=404, detail="PAMAP2 evaluation metrics not found.")
    with open(metrics_path, "r", encoding="utf-8") as file:
        return json.load(file)


@app.get("/api/shl/metrics")
def get_shl_metrics():
    metrics_path = "results/shl_metrics.json"
    if not os.path.exists(metrics_path):
        raise HTTPException(status_code=404, detail="SHL evaluation metrics not found.")
    with open(metrics_path, "r", encoding="utf-8") as file:
        return json.load(file)


@app.post("/api/raw-recordings")
async def import_raw_recording(request: Request):
    """Import and preview raw six-axis CSV; classify when PAMAP2-compatible."""
    content = bytearray()
    async for chunk in request.stream():
        content.extend(chunk)
        if len(content) > MAX_RAW_CSV_BYTES:
            raise HTTPException(
                status_code=413,
                detail=f"CSV exceeds the {MAX_RAW_CSV_BYTES // (1024 * 1024)} MB upload limit.",
            )
    try:
        result = RAW_RECORDINGS.add_csv(bytes(content))
        result["classification_available"] = False
        sample_rate = result["sample_rate_hz"]
        supported_rate = (
            MIN_SUPPORTED_SAMPLE_RATE_HZ
            <= sample_rate
            <= MAX_SUPPORTED_SAMPLE_RATE_HZ
        )
        if PAMAP2_MODEL is not None and supported_rate:
            windows = RAW_RECORDINGS.get_activity_windows(result["recording_id"])
            if len(windows["timestamps"]):
                predictions, confidences = PAMAP2_MODEL.predict(windows["features"])
                timeline = ActivityTimeline.from_predictions(
                    windows["timestamps"],
                    predictions,
                    confidences,
                    session_id=result["recording_id"],
                    window_sec=PAMAP2_TIMELINE_STEP_SEC,
                    merge_gap_sec=PAMAP2_TIMELINE_STEP_SEC,
                    time_origin=0.0,
                    session_duration=windows["recording_duration_sec"],
                )
                TIMELINES_CACHE[result["recording_id"]] = timeline
                CONTEXT_TIMELINES_CACHE[result["recording_id"]] = ContextTimeline.from_labels(
                    windows["timestamps"],
                    {},
                    session_id=result["recording_id"],
                    time_origin=0.0,
                    session_end=timeline.session_end,
                )
                LOCATION_TIMELINES_CACHE[result["recording_id"]] = LocationTimeline.from_labels(
                    windows["timestamps"],
                    {},
                    session_id=result["recording_id"],
                    time_origin=0.0,
                    session_end=timeline.session_end,
                )
                result["classification_available"] = True
                result["classification"] = (
                    "PAMAP2 six-activity model predictions from 5-second chest-IMU windows; "
                    "recording must match training placement, units, and sensor characteristics."
                )
                result["classified_windows"] = len(windows["timestamps"])
            else:
                result["classification"] = (
                    "No continuous 5-second windows at approximately 100 Hz; preview only."
                )
        elif PAMAP2_MODEL is not None:
            result["classification"] = (
                "Preview only; PAMAP2 classification requires approximately 100 Hz "
                f"input (received {sample_rate:.2f} Hz)."
            )
        return result
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error


@app.get("/api/raw-recordings/{recording_id}/signals")
def get_imported_raw_signals(recording_id: str, limit: int = 1500):
    try:
        return RAW_RECORDINGS.get_signals(recording_id, limit=limit)
    except KeyError as error:
        raise HTTPException(status_code=404, detail="Raw recording not found or expired.") from error
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error


@app.get("/api/session/{session_id}/timeline")
def get_timeline(session_id: str):
    """Returns the structured activity timeline and intervals for a session."""
    if session_id not in TIMELINES_CACHE:
        raise HTTPException(status_code=404, detail="Session not found")
    tl = TIMELINES_CACHE[session_id]
    return tl.to_dict()


@app.get("/api/session/{session_id}/context")
def get_context_timeline(session_id: str):
    """Returns annotated context intervals and per-label observation coverage."""
    if session_id not in CONTEXT_TIMELINES_CACHE:
        raise HTTPException(status_code=404, detail="Session context not found")
    return CONTEXT_TIMELINES_CACHE[session_id].to_dict()


@app.get("/api/session/{session_id}/location")
def get_location_timeline(session_id: str):
    """Returns coarse place labels only; raw coordinates and exact addresses are not exposed."""
    if session_id not in LOCATION_TIMELINES_CACHE:
        raise HTTPException(status_code=404, detail="Session place labels not found")
    return LOCATION_TIMELINES_CACHE[session_id].to_dict()


@app.get("/api/session/{session_id}/signals")
def get_sensor_signals(session_id: str, activity: Optional[str] = None, duration_sec: float = 10.0):
    """
    Returns PAMAP2 measured signals for PAMAP2 sessions or a labeled illustration otherwise.
    """
    if session_id in PAMAP2_SESSIONS:
        if PAMAP2_LOADER is None:
            raise HTTPException(status_code=500, detail="PAMAP2 loader not initialized")
        requested_activity = activity or "walking"
        try:
            signals = PAMAP2_LOADER.get_activity_preview(
                PAMAP2_SESSIONS[session_id]["subject"],
                requested_activity,
                duration_sec=min(max(duration_sec, 0.1), 10.0),
            )
        except ValueError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error
        return {
            "session_id": session_id,
            "activity": requested_activity,
            "sampling_rate_hz": signals["sampling_rate_hz"],
            "duration_sec": round(float(signals["timestamps"][-1]), 3),
            "data_source": signals["data_source"],
            "timestamps": signals["timestamps"].tolist(),
            "acc_x": signals["Acc X"].tolist(),
            "acc_y": signals["Acc Y"].tolist(),
            "acc_z": signals["Acc Z"].tolist(),
            "gyro_x": signals["Gyro X"].tolist(),
            "gyro_y": signals["Gyro Y"].tolist(),
            "gyro_z": signals["Gyro Z"].tolist(),
        }

    if session_id in SHL_SESSIONS:
        if SHL_LOADER is None:
            raise HTTPException(status_code=500, detail="SHL loader not initialized.")
        requested_activity = activity or "walking"
        try:
            signals = SHL_LOADER.get_activity_preview(
                SHL_SESSIONS[session_id],
                requested_activity,
                duration_sec=min(max(duration_sec, 0.1), 10.0),
            )
        except ValueError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error
        return {
            "session_id": session_id,
            "activity": requested_activity,
            "sampling_rate_hz": signals["sampling_rate_hz"],
            "duration_sec": round(float(signals["timestamps"][-1]), 3),
            "data_source": signals["data_source"],
            "timestamps": signals["timestamps"].tolist(),
            "acc_x": signals["Acc X"].tolist(),
            "acc_y": signals["Acc Y"].tolist(),
            "acc_z": signals["Acc Z"].tolist(),
            "gyro_x": signals["Gyro X"].tolist(),
            "gyro_y": signals["Gyro Y"].tolist(),
            "gyro_z": signals["Gyro Z"].tolist(),
        }

    if LOADER is None:
        raise HTTPException(status_code=500, detail="Loader not initialized")
    
    act = activity or "walking"
    signals = LOADER.synthesize_25hz_raw_signals(duration_sec=duration_sec, activity=act, sampling_rate=25)
    
    return {
        "session_id": session_id,
        "data_source": "synthetic_illustration",
        "activity": act,
        "sampling_rate_hz": 25,
        "duration_sec": duration_sec,
        "timestamps": signals["time"].tolist(),
        "acc_x": signals["Acc X"].tolist(),
        "acc_y": signals["Acc Y"].tolist(),
        "acc_z": signals["Acc Z"].tolist(),
        "gyro_x": signals["Gyro X"].tolist(),
        "gyro_y": signals["Gyro Y"].tolist(),
        "gyro_z": signals["Gyro Z"].tolist()
    }


@app.post("/api/qa/ask")
def ask_question(req: QuestionRequest):
    """
    Processes natural language question over session timeline:
    Decomposition -> Evidence Retrieval -> Deterministic Math -> Grounded Structured Answer.
    """
    if req.session_id not in TIMELINES_CACHE:
        raise HTTPException(status_code=404, detail="Session not found")

    tl = TIMELINES_CACHE[req.session_id]

    # 1. Question Decomposition
    plan = PLANNER.plan_query(req.question)
    is_raw_imu = (
        req.session_id in PAMAP2_SESSIONS
        or req.session_id in SHL_SESSIONS
        or (req.session_id.startswith("raw-") and req.session_id in TIMELINES_CACHE)
    )
    is_studentlife = req.session_id in STUDENTLIFE_SESSIONS

    if is_raw_imu:
        # PAMAP2 / SHL / uploaded CSV — no context or place labels at all
        supported_activities = (
            set(SHL_LABEL_NAMES)
            if req.session_id in SHL_SESSIONS
            else set(PAMAP2_LABEL_NAMES)
        )
        requested_activities = {
            activity
            for activity in (
                plan.target_activity,
                plan.compare_activity,
                plan.from_activity,
                plan.to_activity,
            )
            if activity is not None
        } | set(plan.target_activities_group)
        if plan.context_label or plan.location_label or plan.from_location or plan.to_location:
            plan.is_supported = False
            plan.operation = "unsupported"
            plan.unsupported_reason = (
                "Raw IMU recordings do not contain context or place labels."
            )
        elif not requested_activities.issubset(supported_activities):
            unsupported = sorted(requested_activities - supported_activities)
            plan.is_supported = False
            plan.operation = "unsupported"
            plan.unsupported_reason = (
                "This selected dataset has no labeled training data for: "
                + ", ".join(unsupported)
                + "."
            )
    elif is_studentlife:
        # StudentLife — has GPS + WiFi location + EMA context, but limited activity taxonomy
        studentlife_supported = {"sitting", "walking", "running"}
        requested_activities = {
            activity
            for activity in (
                plan.target_activity,
                plan.compare_activity,
                plan.from_activity,
                plan.to_activity,
            )
            if activity is not None
        } | set(plan.target_activities_group)
        if requested_activities and not requested_activities.issubset(studentlife_supported):
            unsupported = sorted(requested_activities - studentlife_supported)
            plan.is_supported = False
            plan.operation = "unsupported"
            plan.unsupported_reason = (
                "StudentLife activity sensing maps only sitting, walking, and running. "
                "No labeled data exists for: " + ", ".join(unsupported) + "."
            )

    # 2. Deterministic Execution & Evidence Retrieval
    evidence = ENGINE.execute(
        plan,
        tl,
        context_timeline=CONTEXT_TIMELINES_CACHE.get(req.session_id),
        location_timeline=LOCATION_TIMELINES_CACHE.get(req.session_id),
    )

    # 3. Grounded Answer Assembly
    answer = GENERATOR.generate(plan, evidence)

    return {
        "question": req.question,
        "session_id": req.session_id,
        "plan": plan.to_dict(),
        "evidence": evidence.to_dict(),
        "structured_answer": answer.to_dict(),
        "formatted_text": answer.format_text()
    }


@app.get("/api/benchmark")
def get_benchmark():
    """Returns the full 4-Task Benchmark results and edge deployment profiling."""
    if BENCHMARK_CACHE is None:
        raise HTTPException(status_code=500, detail="Benchmark not ready")
    return BENCHMARK_CACHE


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
