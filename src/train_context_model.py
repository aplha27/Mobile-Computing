"""Train and evaluate the context-conditioned ExtraSensory activity model."""

import argparse
import json
import os

import numpy as np
from sklearn.metrics import accuracy_score, f1_score

from src.context_conditioned_model import evaluate_context_conditioned_model
from src.dataset_loader import ExtraSensoryLoader
from src.sensor_model import SensorUnderstandingPipeline


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train a context/place-conditioned activity model on ExtraSensory."
    )
    parser.add_argument(
        "--zip_path",
        default="ExtraSensory.per_uuid_features_labels.zip",
        help="Path to the ExtraSensory archive.",
    )
    parser.add_argument(
        "--model_path",
        default="checkpoints/context_conditioned_model.pkl",
        help="Where to save the trained model.",
    )
    parser.add_argument(
        "--baseline_path",
        default="checkpoints/sensor_model.pkl",
        help="Existing activity-only model checkpoint for comparison.",
    )
    parser.add_argument(
        "--metrics_path",
        default="results/context_conditioned_model.json",
        help="Where to save held-out evaluation metrics.",
    )
    args = parser.parse_args()

    loader = ExtraSensoryLoader(zip_path=args.zip_path)
    splits = loader.load_dataset_split()
    model, metrics = evaluate_context_conditioned_model(
        splits["train_sessions"], splits["test_sessions"]
    )

    if os.path.exists(args.baseline_path):
        baseline = SensorUnderstandingPipeline()
        baseline.load(args.baseline_path)
        baseline_features = []
        baseline_targets = []
        for session in splits["test_sessions"]:
            valid = np.asarray(session["valid_mask"], dtype=bool)
            if np.any(valid):
                predictions, _ = baseline.predict_raw_features(
                    session["raw_features"][valid]
                )
                baseline_features.append(predictions)
                baseline_targets.append(np.asarray(session["labels"])[valid])
        baseline_predictions = np.concatenate(baseline_features)
        baseline_targets_array = np.concatenate(baseline_targets)
        baseline_macro_f1 = f1_score(
            baseline_targets_array,
            baseline_predictions,
            average="macro",
            zero_division=0,
        )
        baseline_accuracy = accuracy_score(baseline_targets_array, baseline_predictions)
        metrics["activity_only_test_macro_f1"] = float(baseline_macro_f1)
        metrics["activity_only_test_accuracy"] = float(baseline_accuracy)
        metrics["test_macro_f1_delta"] = float(
            metrics["test_macro_f1"] - baseline_macro_f1
        )

    model.save(args.model_path)
    os.makedirs(os.path.dirname(args.metrics_path) or ".", exist_ok=True)
    with open(args.metrics_path, "w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)
    print(json.dumps(metrics, indent=2))
    print(f"Saved context-conditioned model: {args.model_path}")
    print(f"Saved held-out metrics: {args.metrics_path}")


if __name__ == "__main__":
    main()
