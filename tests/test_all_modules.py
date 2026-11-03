"""
test_all_modules.py
===================
Comprehensive unit & integration test suite verifying the Grounded Activity QA system.
"""

import os
import sys
import tempfile
import unittest
import zipfile
import numpy as np
from sklearn.preprocessing import StandardScaler

# Add repo root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.dataset_loader import ExtraSensoryLoader, TARGET_ACTIVITIES, ACTIVITY_DISPLAY_NAMES
from src.sensor_model import SensorUnderstandingPipeline, SensorEncoderNN
from src.timeline_store import ActivityTimeline, ActivityInterval
from src.context_store import ContextTimeline, LocationTimeline
from src.query_planner import QueryPlanner, QueryPlan
from src.deterministic_engine import DeterministicEngine, EvidencePackage
from src.answer_generator import AnswerGenerator, StructuredAnswer
from src.evaluator import BenchmarkEvaluator
from src.context_conditioned_model import encode_annotation_features
from src.raw_recording_store import RawRecordingStore
from src.pamap2_model import (
    PAMAP2_ACTIVITY_MAP,
    PAMAP2ActivityModel,
    make_labeled_windows,
    make_unlabeled_windows,
)
from src.shl_model import (
    SHL_ACTIVITY_MAP,
    SHLActivityModel,
    SHLLoader,
    evaluate_leave_one_subject_out as evaluate_shl_loso,
    make_labeled_windows as make_shl_labeled_windows,
)


class TestGroundedQASystem(unittest.TestCase):

    def setUp(self):
        self.planner = QueryPlanner()
        self.engine = DeterministicEngine()
        self.generator = AnswerGenerator()

        # Mock timeline for testing
        # 0s - 300s: sitting
        # 300s - 900s: walking
        # 900s - 1200s: sitting
        # 1200s - 1500s: running
        # 1500s - 1800s: walking
        # 1800s - 2400s: bicycling
        self.timeline = ActivityTimeline(session_id="test_session")
        self.timeline.session_start = 0.0
        self.timeline.session_end = 2400.0
        self.timeline.total_duration = 2400.0

        self.timeline.intervals = [
            ActivityInterval("sitting", 0, 300, 0.94),
            ActivityInterval("walking", 300, 900, 0.92),
            ActivityInterval("sitting", 900, 1200, 0.96),
            ActivityInterval("running", 1200, 1500, 0.95),
            ActivityInterval("walking", 1500, 1800, 0.91),
            ActivityInterval("bicycling", 1800, 2400, 0.93)
        ]

    def test_query_planner_duration(self):
        plan = self.planner.plan_query("How long did I walk?")
        self.assertEqual(plan.operation, "duration")
        self.assertEqual(plan.target_activity, "walking")
        self.assertTrue(plan.is_supported)

    def test_query_planner_count(self):
        plan = self.planner.plan_query("How many times did the user run?")
        self.assertEqual(plan.operation, "count")
        self.assertEqual(plan.target_activity, "running")
        self.assertTrue(plan.is_supported)

    def test_query_planner_comparison(self):
        plan = self.planner.plan_query("Did the user spend more time walking or sitting?")
        self.assertEqual(plan.operation, "comparison")
        self.assertTrue("walking" in [plan.target_activity, plan.compare_activity])
        self.assertTrue("sitting" in [plan.target_activity, plan.compare_activity])

    def test_query_planner_transition(self):
        plan = self.planner.plan_query("How many times did the user transition from sitting to walking?")
        self.assertEqual(plan.operation, "transition")
        self.assertEqual(plan.from_activity, "sitting")
        self.assertEqual(plan.to_activity, "walking")

    def test_query_planner_open_world(self):
        plan = self.planner.plan_query("Was the user using a wheeled or pedal-based mode of movement?")
        self.assertEqual(plan.operation, "open_world_reasoning")
        self.assertIn("bicycling", plan.target_activities_group)

    def test_query_planner_truthful_rejection(self):
        plan = self.planner.plan_query("What food did the user eat for lunch?")
        self.assertEqual(plan.operation, "unsupported")
        self.assertFalse(plan.is_supported)

    def test_query_planner_context_and_location(self):
        context_plan = self.planner.plan_query("How long was I sitting in a car?")
        self.assertEqual(context_plan.operation, "context_filter")
        self.assertEqual(context_plan.target_activity, "sitting")
        self.assertEqual(context_plan.context_label, "in_a_car")

        location_plan = self.planner.plan_query("How long was I at the gym?")
        self.assertEqual(location_plan.operation, "location_duration")
        self.assertEqual(location_plan.location_label, "gym")

        transition_plan = self.planner.plan_query(
            "How many times did I go from home to work?"
        )
        self.assertEqual(transition_plan.operation, "place_transition")
        self.assertEqual(transition_plan.from_location, "home")
        self.assertEqual(transition_plan.to_location, "work")

    def test_location_data_scope_and_gps_rejection(self):
        plan = self.planner.plan_query("What were my exact GPS coordinates?")
        self.assertEqual(plan.operation, "unsupported")
        self.assertFalse(plan.is_supported)

        distance_plan = self.planner.plan_query("How far did I travel?")
        self.assertEqual(distance_plan.operation, "unsupported")
        self.assertFalse(distance_plan.is_supported)

        coarse_place_plan = self.planner.plan_query(
            "Was I at home according to GPS?"
        )
        self.assertEqual(coarse_place_plan.operation, "context_presence")
        self.assertTrue(coarse_place_plan.is_supported)

        visit_plan = self.planner.plan_query("What places did I visit?")
        self.assertEqual(visit_plan.operation, "location_identification")

    def test_context_model_distinguishes_missing_from_negative_labels(self):
        missing = encode_annotation_features(
            {"indoors": np.array([np.nan])}, {}, row_count=1
        )
        negative = encode_annotation_features(
            {"indoors": np.array([0.0])}, {}, row_count=1
        )
        self.assertEqual(missing[0, 0], 0.0)
        self.assertEqual(missing[0, 1], 0.0)
        self.assertEqual(negative[0, 0], 0.0)
        self.assertEqual(negative[0, 1], 1.0)

    def test_raw_recording_csv_import_is_real_signal_preview_only(self):
        store = RawRecordingStore()
        csv_data = (
            "timestamp,acc_x,acc_y,acc_z,gyro_x,gyro_y,gyro_z\n"
            "0,1,2,3,4,5,6\n"
            "0.04,2,3,4,5,6,7\n"
            "0.08,3,4,5,6,7,8\n"
        ).encode("utf-8")
        imported = store.add_csv(csv_data)
        signals = store.get_signals(imported["recording_id"])
        self.assertEqual(imported["sample_count"], 3)
        self.assertAlmostEqual(imported["sample_rate_hz"], 25.0)
        self.assertEqual(signals["data_source"], "user-provided raw sensor recording")
        self.assertEqual(signals["acc_x"], [1.0, 2.0, 3.0])
        self.assertIn("not performed", imported["classification"])

    def test_raw_recording_csv_rejects_non_monotonic_timestamps(self):
        store = RawRecordingStore()
        csv_data = (
            "timestamp,acc_x,acc_y,acc_z,gyro_x,gyro_y,gyro_z\n"
            "1,1,2,3,4,5,6\n"
            "1,2,3,4,5,6,7\n"
        ).encode("utf-8")
        with self.assertRaisesRegex(ValueError, "strictly increasing"):
            store.add_csv(csv_data)

    def test_sensor_model_checkpoint_preserves_feature_scaler(self):
        scaler = StandardScaler().fit(np.array([[0.0, 2.0], [2.0, 4.0]]))
        model = SensorUnderstandingPipeline(input_dim=2)
        model.feature_scaler = scaler
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = os.path.join(directory, "model.pkl")
            model.save(checkpoint)
            loaded = SensorUnderstandingPipeline()
            loaded.load(checkpoint)
        self.assertIsNotNone(loaded.feature_scaler)
        np.testing.assert_allclose(
            loaded.feature_scaler.transform([[1.0, 3.0]]),
            scaler.transform([[1.0, 3.0]]),
        )

    def test_context_filter_intersects_sensor_and_context_intervals(self):
        timestamps = np.array([0, 60, 120, 180], dtype=float)
        context = ContextTimeline.from_labels(
            timestamps,
            {"in_a_car": np.array([0, 1, 1, 0], dtype=float)},
            session_id="context_session",
        )
        timeline = ActivityTimeline(session_id="context_session")
        timeline.session_end = 240.0
        timeline.intervals = [ActivityInterval("walking", 30, 150, 0.9)]

        plan = self.planner.plan_query("How long did I walk in a car?")
        evidence = self.engine.execute(plan, timeline, context_timeline=context)
        self.assertEqual(evidence.operation, "context_filter")
        self.assertEqual(evidence.result_value, 90.0)
        self.assertEqual(len(evidence.intervals), 1)
        self.assertEqual(evidence.intervals[0].start, 60.0)
        self.assertEqual(evidence.intervals[0].end, 150.0)

    def test_missing_context_is_unknown_not_negative(self):
        timestamps = np.array([0, 60, 120, 180], dtype=float)
        context = ContextTimeline.from_labels(
            timestamps,
            {"in_a_car": np.array([np.nan, np.nan, 0, 0], dtype=float)},
            session_id="partial_context_session",
        )
        timeline = ActivityTimeline(session_id="partial_context_session")
        timeline.session_end = 240.0
        timeline.intervals = [ActivityInterval("walking", 0, 240, 0.9)]

        plan = self.planner.plan_query("Was I walking in a car?")
        evidence = self.engine.execute(plan, timeline, context_timeline=context)
        answer = self.generator.generate(plan, evidence)
        self.assertEqual(evidence.result_value, "Unknown")
        self.assertIn("Unknown", answer.answer)
        self.assertIn("incomplete", answer.answer)

    def test_place_duration_and_transition_use_annotated_labels(self):
        timestamps = np.array([0, 60, 120, 180, 240], dtype=float)
        locations = LocationTimeline.from_labels(
            timestamps,
            {
                "home": np.array([1, 1, 0, 0, 0], dtype=float),
                "work": np.array([0, 0, 1, 1, 0], dtype=float),
                "gym": np.array([np.nan, np.nan, np.nan, np.nan, np.nan]),
            },
            session_id="location_session",
        )
        timeline = ActivityTimeline(session_id="location_session")

        duration_plan = self.planner.plan_query("How long was I at home?")
        duration_evidence = self.engine.execute(
            duration_plan, timeline, location_timeline=locations
        )
        self.assertEqual(duration_evidence.result_value, 120.0)
        duration_answer = self.generator.generate(duration_plan, duration_evidence)
        self.assertEqual(duration_answer.sensor_channels, "N/A (no raw sensor channels)")
        self.assertIn("raw GPS coordinates", duration_answer.explanation)

        transition_plan = self.planner.plan_query(
            "How many times did I go from home to work?"
        )
        transition_evidence = self.engine.execute(
            transition_plan, timeline, location_timeline=locations
        )
        self.assertEqual(transition_evidence.result_value, 1)
        self.assertEqual(
            transition_evidence.metadata["transitions"][0]["transition_time"], 120.0
        )

        missing_plan = self.planner.plan_query("How long was I at the gym?")
        missing_evidence = self.engine.execute(
            missing_plan, timeline, location_timeline=locations
        )
        self.assertFalse(missing_evidence.is_supported)

    def test_deterministic_duration_calculation(self):
        # Walking: (900-300) + (1800-1500) = 600 + 300 = 900 s
        plan = self.planner.plan_query("How long did the user walk?")
        ev = self.engine.execute(plan, self.timeline)
        self.assertEqual(ev.result_value, 900.0)
        self.assertEqual(len(ev.intervals), 2)
        self.assertIn("900 s", ev.calculation_trace)

    def test_deterministic_count_calculation(self):
        # Walking count: 2
        plan = self.planner.plan_query("How many times did the user walk?")
        ev = self.engine.execute(plan, self.timeline)
        self.assertEqual(ev.result_value, 2)

    def test_deterministic_comparison(self):
        # Walking (900s) vs Sitting (300+300 = 600s) -> Walking wins by 300s
        plan = self.planner.plan_query("Did the user spend more time walking or sitting?")
        ev = self.engine.execute(plan, self.timeline)
        self.assertEqual(ev.result_value, "Walking")
        self.assertEqual(ev.metadata["diff"], 300.0)

    def test_deterministic_transition(self):
        # Transitions:
        # sitting(0-300) -> walking(300-900)  [Count 1]
        # walking(300-900) -> sitting(900-1200)
        # sitting(900-1200) -> running(1200-1500)
        # running(1200-1500) -> walking(1500-1800)
        # walking(1500-1800) -> bicycling(1800-2400)
        plan = self.planner.plan_query("How many times did the user transition from sitting to walking?")
        ev = self.engine.execute(plan, self.timeline)
        self.assertEqual(ev.result_value, 1)

    def test_structured_answer_schema(self):
        plan = self.planner.plan_query("How long did the user walk?")
        ev = self.engine.execute(plan, self.timeline)
        ans = self.generator.generate(plan, ev)

        text = ans.format_text()
        self.assertIn("Answer: 900 seconds", text)
        self.assertIn("Activity/Event: Walking", text)
        self.assertIn("Evidence:", text)
        self.assertIn("Timestamp(s): 300–900 s, 1500–1800 s", text)
        self.assertIn("Sensor Modality:\nAccelerometer + Gyroscope", text)
        self.assertIn("Sensor Channel(s):\nAcc X + Acc Y + Acc Z + Gyro X + Gyro Y + Gyro Z", text)
        self.assertIn("Explanation:", text)

    def test_truthful_rejection_structured_output(self):
        plan = self.planner.plan_query("What food did the user eat?")
        ev = self.engine.execute(plan, self.timeline)
        ans = self.generator.generate(plan, ev)

        self.assertFalse(ans.is_supported)
        self.assertIn("Unable to answer", ans.answer)
        self.assertIn("Out-of-scope", ans.activity_event)

    def test_temporal_iou_computation(self):
        # Perfect match
        iou_perfect = self.engine.compute_temporal_iou([(100, 200)], [(100, 200)])
        self.assertAlmostEqual(iou_perfect, 1.0, places=2)

        # 50% overlap: [100, 200] and [150, 250] -> Intersection 50, Union 150 -> IoU = 0.333
        iou_overlap = self.engine.compute_temporal_iou([(100, 200)], [(150, 250)])
        self.assertAlmostEqual(iou_overlap, 50.0 / 150.0, places=2)

        # Disjoint: [100, 200] and [300, 400] -> IoU = 0.0
        iou_disjoint = self.engine.compute_temporal_iou([(100, 200)], [(300, 400)])
        self.assertAlmostEqual(iou_disjoint, 0.0, places=2)

    def test_activity_timeline_supports_explicit_origin_and_recording_duration(self):
        timeline = ActivityTimeline.from_predictions(
            np.array([2.5, 5.0]),
            np.array([0, 1]),
            np.array([0.9, 0.9]),
            window_sec=2.5,
            merge_gap_sec=2.5,
            time_origin=0.0,
            session_duration=8.0,
        )
        self.assertEqual(timeline.session_end, 8.0)
        self.assertEqual(timeline.intervals[0].start, 2.5)
        self.assertEqual(timeline.intervals[-1].end, 7.5)

    def test_kinematic_signal_synthesizer(self):
        loader = ExtraSensoryLoader()
        signals = loader.synthesize_25hz_raw_signals(duration_sec=2.0, activity="walking", sampling_rate=25)
        self.assertEqual(len(signals["time"]), 50)
        self.assertEqual(len(signals["Acc X"]), 50)
        self.assertEqual(len(signals["Gyro Z"]), 50)

    def test_pamap2_windows_keep_only_mapped_stable_activities(self):
        timestamps = np.arange(1000, dtype=float) / 100
        activity = np.full(1000, 4.0)
        activity[750:] = 7.0
        channels = np.column_stack(
            [np.sin(2 * np.pi * timestamps * (index + 1)) for index in range(6)]
        )
        rows = np.column_stack((timestamps, activity, channels))
        windows = make_labeled_windows(rows)
        self.assertEqual(len(windows["labels"]), 2)
        self.assertTrue(np.all(windows["labels"] == 4))
        self.assertEqual(windows["features"].shape, (2, 54))
        self.assertEqual(PAMAP2_ACTIVITY_MAP[4], 4)

    def test_pamap2_unlabeled_inference_requires_approximately_100_hz(self):
        timestamps = np.arange(1000, dtype=float) / 100
        channels = np.ones((1000, 6), dtype=float)
        windows = make_unlabeled_windows(np.column_stack((timestamps, channels)))
        self.assertEqual(windows["features"].shape, (3, 54))
        low_rate = np.arange(1000, dtype=float) / 50
        with self.assertRaisesRegex(ValueError, "approximately 100 Hz"):
            make_unlabeled_windows(np.column_stack((low_rate, channels)))

    def test_pamap2_model_predicts_only_shared_sensorqa_activity_ids(self):
        classes = sorted(PAMAP2_ACTIVITY_MAP.values())
        features = np.vstack(
            [
                np.random.default_rng(class_id).normal(class_id, 0.1, size=(20, 54))
                for class_id in classes
            ]
        )
        labels = np.repeat(classes, 20)
        model = PAMAP2ActivityModel().fit(features, labels)
        predictions, confidence = model.predict(features[:4])
        self.assertEqual(predictions.shape, (4,))
        self.assertEqual(confidence.shape, (4,))
        self.assertTrue(set(predictions).issubset(PAMAP2_ACTIVITY_MAP.values()))

    def test_shl_windows_keep_only_mapped_stable_activities(self):
        timestamps = np.arange(1500, dtype=float) / 100
        labels = np.repeat([2.0, 3.0, 4.0], 500)
        channels = np.column_stack(
            [
                np.sin(2 * np.pi * timestamps * (index + 1))
                for index in range(6)
            ]
        )
        windows = make_shl_labeled_windows(
            np.column_stack((timestamps, labels, channels))
        )
        self.assertEqual(windows["features"].shape, (3, 54))
        self.assertEqual(windows["labels"].tolist(), [4, 5, 6])
        self.assertEqual(SHL_ACTIVITY_MAP, {2: 4, 3: 5, 4: 6})

    def test_shl_loader_streams_aligned_zip_files(self):
        timestamps_ms = 1_500_000_000_000 + np.arange(1000) * 10
        sensor_lines = "\n".join(
            f"{timestamp}.000000 1 2 3 4 5 6"
            for timestamp in timestamps_ms
        )
        label_lines = "\n".join(
            f"{timestamp} 2" for timestamp in timestamps_ms
        )
        with tempfile.TemporaryDirectory() as directory:
            archive_path = os.path.join(directory, "shl-preview.zip")
            with zipfile.ZipFile(archive_path, "w") as archive:
                root = "SHLDataset_preview_v1/User1/220617/"
                archive.writestr(root + "Hand_Motion.txt", sensor_lines)
                archive.writestr(root + "Label.txt", label_lines)
            loader = SHLLoader([archive_path], chunk_size=500)
            descriptors = loader.available_sessions()
            self.assertEqual(len(descriptors), 1)
            session = loader.load_session(descriptors[0])
            self.assertEqual(session["session_id"], "shl-User1-220617")
            self.assertEqual(session["sample_count"], 1000)
            self.assertEqual(session["features"].shape, (3, 54))
            self.assertTrue(np.all(session["labels"] == 4))

    def test_shl_model_predicts_only_shared_activity_ids(self):
        classes = sorted(SHL_ACTIVITY_MAP.values())
        features = np.vstack(
            [
                np.random.default_rng(class_id).normal(
                    class_id, 0.1, size=(20, 54)
                )
                for class_id in classes
            ]
        )
        labels = np.repeat(classes, 20)
        model = SHLActivityModel().fit(features, labels)
        predictions, confidence = model.predict(features[:4])
        self.assertEqual(predictions.shape, (4,))
        self.assertEqual(confidence.shape, (4,))
        self.assertTrue(set(predictions).issubset(classes))

    def test_shl_loso_holds_out_all_sessions_for_each_subject(self):
        classes = sorted(SHL_ACTIVITY_MAP.values())
        sessions = []
        for subject in ("User1", "User2", "User3"):
            for day in ("220617", "260617"):
                features = np.vstack(
                    [
                        np.random.default_rng(class_id).normal(
                            class_id, 0.1, size=(10, 54)
                        )
                        for class_id in classes
                    ]
                )
                sessions.append(
                    {
                        "subject": subject,
                        "session_id": f"shl-{subject}-{day}",
                        "features": features,
                        "labels": np.repeat(classes, 10),
                    }
                )
        evaluation = evaluate_shl_loso(sessions, include_session_predictions=True)
        metrics = evaluation["metrics"]
        self.assertEqual(metrics["subject_count"], 3)
        self.assertEqual(metrics["windows"], 180)
        self.assertEqual(len(evaluation["session_predictions"]), 6)


if __name__ == "__main__":
    unittest.main()
