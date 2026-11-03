"""Train and participant-held-out evaluate the raw-signal PAMAP2 classifier."""

import argparse
import json
import os

import numpy as np

from src.pamap2_model import (
    PAMAP2ActivityModel,
    PAMAP2Loader,
    evaluate_leave_one_subject_out,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build SensorQA's raw-signal PAMAP2 activity model."
    )
    parser.add_argument("--archive", default="pamap2+physical+activity+monitoring.zip")
    parser.add_argument("--model", default="checkpoints/pamap2_activity_model.pkl")
    parser.add_argument("--metrics", default="results/pamap2_metrics.json")
    args = parser.parse_args()

    loader = PAMAP2Loader(args.archive)
    sessions = loader.load_protocol_subjects()
    usable_sessions = [session for session in sessions if len(session["labels"])]
    if len(usable_sessions) < 2:
        raise ValueError("PAMAP2 archive did not contain enough labeled protocol subjects.")

    for session in usable_sessions:
        print(
            f"{session['subject']}: {len(session['labels'])} windows, "
            f"{len(np.unique(session['labels']))} mapped activities",
            flush=True,
        )

    metrics = evaluate_leave_one_subject_out(usable_sessions)
    training_features = np.vstack([session["features"] for session in usable_sessions])
    training_labels = np.concatenate([session["labels"] for session in usable_sessions])
    model = PAMAP2ActivityModel().fit(training_features, training_labels)
    model.save(args.model)

    os.makedirs(os.path.dirname(args.metrics) or ".", exist_ok=True)
    with open(args.metrics, "w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)
    print(json.dumps(metrics, indent=2))
    print(f"Saved PAMAP2 raw-signal model: {args.model}")
    print(f"Saved participant-held-out metrics: {args.metrics}")


if __name__ == "__main__":
    main()
