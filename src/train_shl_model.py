"""Train and participant-held-out evaluate the SHL Hand-placement classifier."""

import argparse
import json
import os
import pickle
from typing import Any, Dict

import numpy as np

from src.shl_model import (
    SHL_ACTIVITY_MAP,
    SHL_ARCHIVES,
    SHLActivityModel,
    SHLLoader,
    evaluate_leave_one_subject_out,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build SensorQA's separate SHL Hand-IMU activity model."
    )
    parser.add_argument("--archives", nargs="+", default=list(SHL_ARCHIVES))
    parser.add_argument("--model", default="checkpoints/shl_activity_model.pkl")
    parser.add_argument("--metrics", default="results/shl_metrics.json")
    parser.add_argument(
        "--predictions",
        default="results/shl_session_predictions.pkl",
        help="Save in-sample timeline predictions for the local web app.",
    )
    args = parser.parse_args()

    loader = SHLLoader(args.archives)
    sessions = loader.load_sessions()
    usable_sessions = [session for session in sessions if len(session["labels"])]
    participants = {session["subject"] for session in usable_sessions}
    if len(participants) < 2:
        raise ValueError(
            "SHL preview archives must provide labeled windows from at least two users."
        )

    for session in sessions:
        print(
            f"{session['session_id']}: {session['sample_count']} samples, "
            f"{len(session['labels'])} mapped windows, "
            f"{len(np.unique(session['labels']))} mapped activities",
            flush=True,
        )

    evaluation = evaluate_leave_one_subject_out(
        usable_sessions, include_session_predictions=True
    )
    metrics = evaluation["metrics"]
    out_of_fold_predictions = evaluation["session_predictions"]
    training_features = np.vstack(
        [session["features"] for session in usable_sessions]
    )
    training_labels = np.concatenate(
        [session["labels"] for session in usable_sessions]
    )
    model = SHLActivityModel().fit(training_features, training_labels)
    model.save(args.model)

    prediction_cache: Dict[str, Any] = {
        "version": 1,
        "activity_map": SHL_ACTIVITY_MAP,
        "sessions": {},
    }
    for session in usable_sessions:
        session_prediction = out_of_fold_predictions[session["session_id"]]
        prediction_cache["sessions"][session["session_id"]] = {
            "subject": session["subject"],
            "date": session["date"],
            "archive_path": session["archive_path"],
            "recording_start": session["recording_start"],
            "recording_duration_sec": session["recording_duration_sec"],
            "sample_rate_hz": session["sample_rate_hz"],
            "timestamps": session["timestamps"],
            "predictions": session_prediction["predictions"],
            "confidences": session_prediction["confidences"],
        }

    os.makedirs(os.path.dirname(args.metrics) or ".", exist_ok=True)
    with open(args.metrics, "w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)
    os.makedirs(os.path.dirname(args.predictions) or ".", exist_ok=True)
    with open(args.predictions, "wb") as file:
        pickle.dump(prediction_cache, file)

    print(json.dumps(metrics, indent=2))
    print(f"Saved SHL Hand-IMU model: {args.model}")
    print(f"Saved participant-held-out metrics: {args.metrics}")
    print(f"Saved session timeline predictions: {args.predictions}")


if __name__ == "__main__":
    main()
