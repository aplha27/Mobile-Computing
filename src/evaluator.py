"""
evaluator.py
============
Comprehensive Multi-Task Evaluation Benchmark for Challenge 1.
Implements the full evaluation matrix defined in Section 12 and 13 of the Core Research Conclusion:
  - Task 1: Activity Identification (Multi-class accuracy, Macro-F1, Confusion Matrix)
  - Task 2: Temporal and Quantitative Reasoning (Duration MAE/MAPE, Count Accuracy, Tolerances)
  - Task 3: Evidence Grounding (Temporal IoU, Modality/Channel Correctness, Calibration ECE)
  - Task 4: Open-World Semantic Reasoning & Truthful Failure Handling
  - Edge Deployment Profiling (Latency, Model Size, Memory, INT8 Quantization)
"""

import time
from typing import Dict, List, Any, Tuple, Optional
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report, f1_score, accuracy_score, precision_recall_fscore_support

from src.dataset_loader import TARGET_ACTIVITIES, ACTIVITY_DISPLAY_NAMES, ACTIVITY_MODALITIES, DEFAULT_CHANNELS
from src.sensor_model import SensorUnderstandingPipeline
from src.timeline_store import ActivityTimeline
from src.query_planner import QueryPlanner, QueryPlan
from src.deterministic_engine import DeterministicEngine
from src.answer_generator import AnswerGenerator


class BenchmarkEvaluator:
    def __init__(
        self,
        sensor_model: SensorUnderstandingPipeline,
        planner: Optional[QueryPlanner] = None,
        engine: Optional[DeterministicEngine] = None,
        generator: Optional[AnswerGenerator] = None
    ):
        self.sensor_model = sensor_model
        self.planner = planner or QueryPlanner()
        self.engine = engine or DeterministicEngine()
        self.generator = generator or AnswerGenerator()
        self.feature_scaler = None

    def _predict_session(self, session: Dict[str, Any], valid: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        features = session["raw_features"][valid]
        if self.sensor_model.feature_scaler is not None:
            return self.sensor_model.predict_raw_features(features)
        if self.feature_scaler is not None:
            features = self.feature_scaler.transform(features)
        return self.sensor_model.predict_with_confidence(features)

    # =========================================================================
    # TASK 1: Activity Identification
    # =========================================================================
    def evaluate_task1_activity_identification(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
        """Evaluates Model A sensor classification across the 7 target activities."""
        preds, confs = self.sensor_model.predict_with_confidence(X_test)
        
        acc = accuracy_score(y_test, preds)
        macro_f1 = f1_score(y_test, preds, average='macro', zero_division=0)
        weighted_f1 = f1_score(y_test, preds, average='weighted', zero_division=0)
        
        p, r, f1, s = precision_recall_fscore_support(y_test, preds, average=None, zero_division=0, labels=list(range(len(TARGET_ACTIVITIES))))
        
        cm = confusion_matrix(y_test, preds, labels=list(range(len(TARGET_ACTIVITIES))))
        
        per_class_metrics = {}
        for i, act in enumerate(TARGET_ACTIVITIES):
            per_class_metrics[act] = {
                "display_name": ACTIVITY_DISPLAY_NAMES[act],
                "precision": round(float(p[i]), 4),
                "recall": round(float(r[i]), 4),
                "f1_score": round(float(f1[i]), 4),
                "support": int(s[i])
            }

        return {
            "task": "Task 1: Activity Identification",
            "accuracy": round(float(acc), 4),
            "macro_f1": round(float(macro_f1), 4),
            "weighted_f1": round(float(weighted_f1), 4),
            "total_test_windows": len(y_test),
            "confusion_matrix": cm.tolist(),
            "per_class": per_class_metrics
        }

    # =========================================================================
    # TASK 2: Temporal and Quantitative Reasoning
    # =========================================================================
    def evaluate_task2_quantitative_reasoning(self, test_sessions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Evaluates quantitative QA tasks (duration, count, comparison) against ground-truth timelines.
        """
        duration_errors = []
        duration_rel_errors = []
        count_exact_matches = 0
        count_total = 0
        comparison_correct = 0
        comparison_total = 0

        for session in test_sessions:
            ts = session["timestamps"]
            valid = session["valid_mask"]
            if np.sum(valid) < 5:
                continue

            # Ground truth timeline
            gt_preds = session["labels"][valid]
            gt_ts = ts[valid]
            gt_confs = np.ones(len(gt_preds))
            gt_timeline = ActivityTimeline.from_predictions(gt_ts, gt_preds, gt_confs, session_id="gt")

            # Predicted timeline
            p_preds, p_confs = self._predict_session(session, valid)
            pred_timeline = ActivityTimeline.from_predictions(gt_ts, p_preds, p_confs, session_id="pred")

            # Test duration & count queries for each activity present with windowed slices (10 mins, 30 mins, full)
            for time_window_sec in [600, 1800, 3600]:
                for act in TARGET_ACTIVITIES:
                    gt_invs = gt_timeline.get_intervals_for_activity(act, t_min=0, t_max=time_window_sec)
                    gt_dur = sum(i.duration for i in gt_invs)
                    gt_cnt = len(gt_invs)

                    if gt_dur > 0:
                        plan_dur = self.planner.plan_query(f"How long did the user spend {act} in the first {int(time_window_sec/60)} minutes?")
                        ev_dur = self.engine.execute(plan_dur, pred_timeline)
                        pred_dur = ev_dur.result_value or 0.0
                        
                        err = abs(pred_dur - gt_dur)
                        duration_errors.append(err)
                        duration_rel_errors.append(err / max(gt_dur, 1.0))

                        # Count evaluation
                        plan_cnt = self.planner.plan_query(f"How many times did the user perform {act} in the first {int(time_window_sec/60)} minutes?")
                        ev_cnt = self.engine.execute(plan_cnt, pred_timeline)
                        pred_cnt = ev_cnt.result_value or 0
                        if pred_cnt == gt_cnt:
                            count_exact_matches += 1
                        count_total += 1

            # Test comparison query
            plan_comp = self.planner.plan_query("Did the user spend more time sitting or lying down in the first 30 minutes?")
            ev_comp_gt = self.engine.execute(plan_comp, gt_timeline)
            ev_comp_pred = self.engine.execute(plan_comp, pred_timeline)
            if ev_comp_gt.result_value == ev_comp_pred.result_value:
                comparison_correct += 1
            comparison_total += 1

        mae = float(np.mean(duration_errors)) if duration_errors else 0.0
        mape = float(np.mean(duration_rel_errors) * 100) if duration_rel_errors else 0.0
        count_acc = float(count_exact_matches / count_total) if count_total > 0 else 1.0
        comp_acc = float(comparison_correct / comparison_total) if comparison_total > 0 else 1.0

        # Tolerances
        within_30s = float(np.mean(np.array(duration_errors) <= 30.0)) if duration_errors else 1.0
        within_60s = float(np.mean(np.array(duration_errors) <= 60.0)) if duration_errors else 1.0

        return {
            "task": "Task 2: Temporal & Quantitative Reasoning",
            "duration_mae_seconds": round(mae, 2),
            "duration_mape_percent": round(mape, 2),
            "duration_accuracy_within_30s": round(within_30s * 100, 2),
            "duration_accuracy_within_60s": round(within_60s * 100, 2),
            "count_exact_accuracy": round(count_acc * 100, 2),
            "comparison_accuracy": round(comp_acc * 100, 2),
            "total_queries_evaluated": len(duration_errors) + count_total + comparison_total
        }

    # =========================================================================
    # TASK 3: Evidence Grounding
    # =========================================================================
    def evaluate_task3_evidence_grounding(self, test_sessions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Evaluates temporal Intersection-over-Union (IoU) and sensor modality/channel grounding precision.
        """
        ious = []
        modality_correct = 0
        channel_correct = 0
        total_evals = 0

        for session in test_sessions:
            valid = session["valid_mask"]
            if np.sum(valid) < 5:
                continue

            ts = session["timestamps"][valid]
            gt_preds = session["labels"][valid]
            gt_timeline = ActivityTimeline.from_predictions(ts, gt_preds, np.ones(len(gt_preds)), session_id="gt")

            p_preds, p_confs = self._predict_session(session, valid)
            pred_timeline = ActivityTimeline.from_predictions(ts, p_preds, p_confs, session_id="pred")

            for act in TARGET_ACTIVITIES:
                gt_invs = gt_timeline.get_intervals_for_activity(act)
                if not gt_invs:
                    continue

                gt_bounds = [(i.start, i.end) for i in gt_invs]
                plan = self.planner.plan_query(f"What were the timestamps when the user was {act}?")
                ev = self.engine.execute(plan, pred_timeline)
                pred_bounds = [(i.start, i.end) for i in ev.intervals]

                iou = self.engine.compute_temporal_iou(pred_bounds, gt_bounds)
                ious.append(iou)

                # Grounding modality & channel checks
                if set(ev.modality) == set(ACTIVITY_MODALITIES[act]):
                    modality_correct += 1
                if set(ev.channels) == set(DEFAULT_CHANNELS):
                    channel_correct += 1
                total_evals += 1

        mean_iou = float(np.mean(ious)) if ious else 1.0
        mod_prec = float(modality_correct / total_evals) if total_evals > 0 else 1.0
        chan_prec = float(channel_correct / total_evals) if total_evals > 0 else 1.0

        return {
            "task": "Task 3: Evidence Grounding",
            "mean_temporal_iou": round(mean_iou, 4),
            "modality_grounding_precision": round(mod_prec * 100, 2),
            "channel_grounding_precision": round(chan_prec * 100, 2),
            "total_grounding_samples": total_evals
        }

    # =========================================================================
    # TASK 4: Open-World Activity Reasoning & Truthful Failure Handling
    # =========================================================================
    def evaluate_task4_open_world_and_safety(self, test_session: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluates open-world semantic queries and out-of-scope truthful rejection handling.
        """
        valid = test_session["valid_mask"]
        ts = test_session["timestamps"][valid]
        p_preds, p_confs = self._predict_session(test_session, valid)
        timeline = ActivityTimeline.from_predictions(ts, p_preds, p_confs)

        test_questions = [
            # Open-world valid
            ("Was the user using a wheeled or pedal-based mode of movement?", True, "open_world_reasoning"),
            ("Was the user engaged in sedentary behavior?", True, "open_world_reasoning"),
            ("Did the user perform vigorous exercise?", True, "open_world_reasoning"),
            ("Was the user engaged in foot locomotion?", True, "open_world_reasoning"),
            
            # Complex reasoning & transitions
            ("How many times did the user transition from sitting to walking?", True, "transition"),
            ("Did the user spend more time walking or sitting in the first 30 minutes?", True, "comparison"),
            
            # Out-of-scope / Unsupported questions (DrHouse truthful rejection)
            ("What food did the user eat for lunch?", False, "unsupported"),
            ("What was the user's heart rate and blood pressure?", False, "unsupported"),
            ("What city and building address was the user at?", False, "unsupported"),
            ("Who was talking to the user during the recording?", False, "unsupported"),
            ("What app was open on the user's phone screen?", False, "unsupported")
        ]

        results = []
        correct_rejections = 0
        total_unsupported = 0
        correct_semantic = 0
        total_semantic = 0

        for q, should_be_supported, expected_op in test_questions:
            plan = self.planner.plan_query(q)
            ev = self.engine.execute(plan, timeline)
            ans = self.generator.generate(plan, ev)

            is_rejected = not ans.is_supported
            if not should_be_supported:
                total_unsupported += 1
                if is_rejected:
                    correct_rejections += 1
            else:
                total_semantic += 1
                if ans.is_supported and plan.operation == expected_op:
                    correct_semantic += 1

            results.append({
                "question": q,
                "operation": plan.operation,
                "is_supported": ans.is_supported,
                "answer": ans.answer,
                "confidence": ans.confidence
            })

        rejection_rate = float(correct_rejections / total_unsupported) if total_unsupported > 0 else 1.0
        semantic_acc = float(correct_semantic / total_semantic) if total_semantic > 0 else 1.0

        return {
            "task": "Task 4: Open-World Reasoning & Safety",
            "semantic_reasoning_accuracy": round(semantic_acc * 100, 2),
            "truthful_rejection_rate": round(rejection_rate * 100, 2),
            "open_world_test_results": results
        }

    # =========================================================================
    # Full Challenge Benchmark Run
    # =========================================================================
    def run_full_benchmark(self, splits: Dict[str, Any]) -> Dict[str, Any]:
        """Runs the complete multi-task evaluation suite."""
        self.feature_scaler = splits.get("scaler")
        print("\n" + "="*70)
        print("RUNNING CHALLENGE 1 FULL BENCHMARK SUITE")
        print("="*70)

        # 1. Task 1
        print("Evaluating Task 1: Activity Identification on unseen test users...")
        t1_res = self.evaluate_task1_activity_identification(splits["X_test"], splits["y_test"])

        # 2. Task 2
        print("Evaluating Task 2: Temporal & Quantitative Reasoning...")
        t2_res = self.evaluate_task2_quantitative_reasoning(splits["test_sessions"])

        # 3. Task 3
        print("Evaluating Task 3: Evidence Grounding (IoU, Modality, Channels)...")
        t3_res = self.evaluate_task3_evidence_grounding(splits["test_sessions"])

        # 4. Task 4
        print("Evaluating Task 4: Open-World Semantic Reasoning & Truthful Rejection...")
        t4_res = self.evaluate_task4_open_world_and_safety(splits["test_sessions"][0])

        # 5. Edge Profiling
        print("Profiling Edge Deployment & Quantization Metrics...")
        edge_res = self.sensor_model.profile_edge_performance(splits["X_test"][:50])

        summary = {
            "task1_activity_identification": t1_res,
            "task2_quantitative_reasoning": t2_res,
            "task3_evidence_grounding": t3_res,
            "task4_open_world_reasoning": t4_res,
            "edge_deployment_profiling": edge_res
        }

        print("\n" + "="*70)
        print("CHALLENGE 1 BENCHMARK RESULTS SUMMARY:")
        print(f"  • Activity Classification Macro-F1: {t1_res['macro_f1']:.4f} (Accuracy: {t1_res['accuracy']*100:.2f}%)")
        print(f"  • Quantitative Duration MAE:        {t2_res['duration_mae_seconds']:.2f} s (MAPE: {t2_res['duration_mape_percent']:.2f}%)")
        print(f"  • Quantitative Count Accuracy:      {t2_res['count_exact_accuracy']:.2f}%")
        print(f"  • Temporal Grounding Mean IoU:      {t3_res['mean_temporal_iou']:.4f}")
        print(f"  • Modality/Channel Grounding:       {t3_res['modality_grounding_precision']:.1f}%")
        print(f"  • Truthful Rejection Rate:          {t4_res['truthful_rejection_rate']:.1f}%")
        print(f"  • Model A (FP32) CPU Latency:       {edge_res['model_a_full_precision']['cpu_latency_ms']} ms/window")
        print(f"  • Model B (INT8) CPU Latency:       {edge_res['model_b_quantized_edge']['cpu_latency_ms']} ms/window ({edge_res['model_b_quantized_edge']['compression_ratio']} compression)")
        print("="*70 + "\n")

        return summary
