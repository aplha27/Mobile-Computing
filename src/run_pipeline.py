"""
run_pipeline.py
===============
End-to-end CLI Pipeline Runner.
Loads ExtraSensory dataset, trains Model A (Sensor Understanding Pipeline),
builds Activity Timelines, executes QA queries, runs the full 4-Task Benchmark Suite,
and exports evaluation metrics and checkpoints.
"""

import os
import sys
import json
import argparse
import numpy as np

# Ensure immediate stdout flushing
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(line_buffering=True)

# Add repo root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.dataset_loader import ExtraSensoryLoader, TARGET_ACTIVITIES, ACTIVITY_DISPLAY_NAMES
from src.sensor_model import SensorUnderstandingPipeline
from src.context_conditioned_model import ContextConditionedActivityModel
from src.timeline_store import ActivityTimeline
from src.context_store import ContextTimeline, LocationTimeline
from src.query_planner import QueryPlanner
from src.deterministic_engine import DeterministicEngine
from src.answer_generator import AnswerGenerator
from src.evaluator import BenchmarkEvaluator


def main():
    parser = argparse.ArgumentParser(description="Challenge 1: Grounded Activity QA Pipeline")
    parser.add_argument("--mode", type=str, default="benchmark", choices=["train", "benchmark", "demo", "query"],
                        help="Execution mode: train, benchmark, demo, query")
    parser.add_argument("--zip_path", type=str, default="ExtraSensory.per_uuid_features_labels.zip",
                        help="Path to ExtraSensory zip dataset")
    parser.add_argument("--model_save_path", type=str, default="checkpoints/sensor_model.pkl",
                        help="Path to save/load model checkpoint")
    parser.add_argument("--context_model_path", type=str, default="checkpoints/context_conditioned_model.pkl",
                        help="Optional context-conditioned model checkpoint")
    parser.add_argument("--train_users", type=int, default=15, help="Number of training users")
    parser.add_argument("--val_users", type=int, default=4, help="Number of validation users")
    parser.add_argument("--test_users", type=int, default=4, help="Number of test users")
    parser.add_argument("--question", type=str, default="How long did the user spend walking?",
                        help="Question to query (used with --mode query)")
    args = parser.parse_args()

    print("="*75)
    print("CHALLENGE 1: GROUNDED ACTIVITY QUESTION-ANSWERING OVER WEARABLE SENSORS")
    print("ExtraSensory precomputed features + observed context/place labels | Sensor2Text + SensorChat + DrHouse + JARVIS")
    print("="*75)

    loader = ExtraSensoryLoader(zip_path=args.zip_path)
    splits = loader.load_dataset_split(
        num_train_users=args.train_users,
        num_val_users=args.val_users,
        num_test_users=args.test_users
    )

    sensor_model = SensorUnderstandingPipeline()

    if os.path.exists(args.model_save_path) and args.mode != "train":
        print(f"Loading existing checkpoint from: {args.model_save_path}")
        sensor_model.load(args.model_save_path)
        if sensor_model.feature_scaler is None:
            print("Checkpoint has no saved feature scaler; retraining pipeline...")
            sensor_model.fit_full_pipeline(splits)
            sensor_model.save(args.model_save_path)
    else:
        train_res = sensor_model.fit_full_pipeline(splits)
        sensor_model.save(args.model_save_path)
        print(f"Model saved to {args.model_save_path}")

    context_model = (
        ContextConditionedActivityModel.load(args.context_model_path)
        if os.path.exists(args.context_model_path)
        else None
    )

    planner = QueryPlanner()
    engine = DeterministicEngine()
    generator = AnswerGenerator()

    if args.mode in ["benchmark", "train"]:
        evaluator = BenchmarkEvaluator(sensor_model, planner, engine, generator)
        benchmark_results = evaluator.run_full_benchmark(splits)
        
        # Save results to json
        os.makedirs("results", exist_ok=True)
        with open("results/benchmark_results.json", "w") as f:
            json.dump(benchmark_results, f, indent=2)
        print("Detailed benchmark metrics exported to: results/benchmark_results.json")

    # Interactive sample demonstration
    test_session = splits["test_sessions"][0]
    valid = test_session["valid_mask"]
    ts = test_session["timestamps"][valid]
    if context_model is not None:
        preds, confs = context_model.predict(
            test_session["raw_features"][valid],
            {label: values[valid] for label, values in test_session["context_labels"].items()},
            {label: values[valid] for label, values in test_session["location_labels"].items()},
        )
    else:
        preds, confs = sensor_model.predict_raw_features(
            test_session["raw_features"][valid]
        )
    timeline = ActivityTimeline.from_predictions(ts, preds, confs, session_id=test_session["uuid"])
    context_timeline = ContextTimeline.from_labels(
        test_session["timestamps"],
        test_session["context_labels"],
        session_id=test_session["uuid"],
        time_origin=float(ts[0]) if len(ts) else None,
        session_end=timeline.session_end,
    )
    location_timeline = LocationTimeline.from_labels(
        test_session["timestamps"],
        test_session["location_labels"],
        session_id=test_session["uuid"],
        time_origin=float(ts[0]) if len(ts) else None,
        session_end=timeline.session_end,
    )

    demo_questions = [
        "How long did the user walk?",
        "How many times did the user perform bicycling?",
        "Did the user spend more time sitting or lying down?",
        "When did walking begin?",
        "How many times did the user transition from sitting to walking?",
        "Was the user using a wheeled or pedal-based mode of movement?",
        "How long was I sitting in a car?",
        "How long was I at the gym?",
        "Did I walk while at work?",
        "How many times did I go from home to work?",
        "What meal was the user eating?"  # Unsupported / truthful rejection
    ] if args.mode != "query" else [args.question]

    print("\n" + "="*75)
    print(f"SAMPLE GROUNDED QA INFERENCE ON UNSEEN USER ({test_session['uuid'][:8]}...)")
    print("="*75)

    for q in demo_questions:
        print(f"\n[USER QUESTION]: \"{q}\"")
        plan = planner.plan_query(q)
        print(f"[DECOMPOSED PLAN]: op='{plan.operation}', target='{plan.target_activity}', compare='{plan.compare_activity}', supported={plan.is_supported}")
        ev = engine.execute(
            plan,
            timeline,
            context_timeline=context_timeline,
            location_timeline=location_timeline,
        )
        print(f"[DETERMINISTIC TRACE]: {ev.calculation_trace}")
        ans = generator.generate(plan, ev)
        print("-" * 50)
        print(ans.format_text())
        print("-" * 50)


if __name__ == "__main__":
    main()
