"""SHL hand-placement raw-signal windowing, training, and evaluation."""

import os
import pickle
import re
import zipfile
from itertools import zip_longest
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

from src.dataset_loader import TARGET_ACTIVITIES
from src.pamap2_model import (
    extract_window_features,
    MAX_CONTIGUOUS_SAMPLE_GAP_SEC,
    WINDOW_HOP,
    WINDOW_SAMPLES,
    WINDOW_SEC,
)


SHL_ARCHIVES = (
    "SHLDataset_preview_v1_part1.zip",
    "SHLDataset_preview_v1_part2.zip",
    "SHLDataset_preview_v1_part3.zip",
)
SHL_ACTIVITY_MAP = {
    2: 4,  # walking
    3: 5,  # running
    4: 6,  # bicycling
}
SHL_RAW_ACTIVITY_NAMES = {
    0: "unlabeled",
    1: "still",
    2: "walking",
    3: "running",
    4: "bicycling",
    5: "car",
    6: "bus",
    7: "train",
    8: "subway",
}
SHL_LABEL_NAMES = {
    TARGET_ACTIVITIES[target]: SHL_RAW_ACTIVITY_NAMES[source]
    for source, target in SHL_ACTIVITY_MAP.items()
}
SHL_SENSOR_COLUMNS = ["Acc X", "Acc Y", "Acc Z", "Gyro X", "Gyro Y", "Gyro Z"]
SHL_SAMPLE_RATE_HZ = 100.0
SHL_MIN_SAMPLE_RATE_HZ = 90.0
SHL_MAX_SAMPLE_RATE_HZ = 110.0
SHL_WINDOW_SEC = WINDOW_SEC
SHL_WINDOW_HOP_SEC = 2.5
SHL_TIMELINE_STEP_SEC = SHL_WINDOW_HOP_SEC
SHL_CHUNK_SIZE = 200_000
SHL_CHECKPOINT_VERSION = 1
SHL_ROOT = "SHLDataset_preview_v1"


def _mapped_label(raw_label: float) -> int:
    if not np.isfinite(raw_label):
        return -1
    return SHL_ACTIVITY_MAP.get(int(raw_label), -1)


def _valid_window_positions(rows: np.ndarray, starts: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Return starts and mapped labels for stable, finite, contiguous windows."""
    if not len(starts):
        return np.empty(0, dtype=int), np.empty(0, dtype=int)

    raw_labels = rows[:, 1]
    label_changes = np.concatenate(
        ([0], np.cumsum(raw_labels[1:] != raw_labels[:-1]))
    )
    labels_stable = (
        label_changes[starts + WINDOW_SAMPLES - 1] == label_changes[starts]
    )
    mapped = np.asarray([_mapped_label(value) for value in raw_labels[starts]])
    finite_rows = np.isfinite(rows[:, 2:]).all(axis=1)
    invalid_prefix = np.concatenate(
        ([0], np.cumsum(~finite_rows, dtype=np.int64))
    )
    finite_windows = (
        invalid_prefix[starts + WINDOW_SAMPLES]
        == invalid_prefix[starts]
    )

    timestamps = rows[:, 0]
    gaps = np.diff(timestamps) > MAX_CONTIGUOUS_SAMPLE_GAP_SEC
    gap_prefix = np.concatenate(([0], np.cumsum(gaps, dtype=np.int64)))
    contiguous_windows = (
        gap_prefix[starts + WINDOW_SAMPLES - 1] == gap_prefix[starts]
    )

    keep = (mapped >= 0) & labels_stable & finite_windows & contiguous_windows
    return starts[keep], mapped[keep]


def make_labeled_windows(
    rows: np.ndarray,
    window_samples: int = WINDOW_SAMPLES,
    hop_samples: int = WINDOW_HOP,
) -> Dict[str, np.ndarray]:
    """Create five-second windows from timestamp, SHL main label, and six IMU channels."""
    rows = np.asarray(rows, dtype=float)
    if rows.ndim != 2 or rows.shape[1] != 8:
        raise ValueError("SHL rows must contain timestamp, label, and 6 sensor channels.")
    if window_samples != WINDOW_SAMPLES or hop_samples != WINDOW_HOP:
        raise ValueError("SHL windows use the fixed five-second/2.5-second protocol.")
    if len(rows) < WINDOW_SAMPLES:
        return {
            "timestamps": np.empty(0),
            "features": np.empty((0, 54), dtype=np.float32),
            "labels": np.empty(0, dtype=int),
        }

    starts = np.arange(0, len(rows) - WINDOW_SAMPLES + 1, WINDOW_HOP)
    starts, labels = _valid_window_positions(rows, starts)
    if not len(starts):
        return {
            "timestamps": np.empty(0),
            "features": np.empty((0, 54), dtype=np.float32),
            "labels": np.empty(0, dtype=int),
        }
    windows = np.stack(
        [rows[start : start + WINDOW_SAMPLES, 2:] for start in starts]
    )
    return {
        "timestamps": rows[starts, 0],
        "features": extract_window_features(windows),
        "labels": labels,
    }


class SHLLoader:
    """Streams SHL hand-IMU sessions and their timestamp-aligned labels from ZIP files."""

    def __init__(
        self,
        archive_paths: Sequence[str] = SHL_ARCHIVES,
        chunk_size: int = SHL_CHUNK_SIZE,
    ) -> None:
        if chunk_size < WINDOW_SAMPLES:
            raise ValueError("SHL chunk size must hold at least one model window.")
        self.archive_paths = tuple(archive_paths)
        self.chunk_size = chunk_size

    def available_sessions(self) -> List[Dict[str, str]]:
        sessions: List[Dict[str, str]] = []
        for archive_path in self.archive_paths:
            if not os.path.isfile(archive_path):
                raise FileNotFoundError(f"SHL archive not found: {archive_path}")
            with zipfile.ZipFile(archive_path) as archive:
                members = set(archive.namelist())
            for member in sorted(members):
                match = re.search(
                    rf"(?:^|/){re.escape(SHL_ROOT)}/(User\d+)/(\d{{6}})/Hand_Motion\.txt$",
                    member,
                )
                if not match:
                    continue
                subject, date = match.groups()
                prefix = member.rsplit("/", 1)[0]
                label_member = f"{prefix}/Label.txt"
                if label_member not in members:
                    raise ValueError(f"SHL session is missing labels: {label_member}")
                sessions.append(
                    {
                        "archive_path": archive_path,
                        "subject": subject,
                        "date": date,
                        "session_id": f"shl-{subject}-{date}",
                        "motion_member": member,
                        "label_member": label_member,
                    }
                )
        return sessions

    def load_session(self, session: Dict[str, str]) -> Dict[str, Any]:
        """Read one session in bounded chunks and extract mapped activity windows."""
        timestamps: List[np.ndarray] = []
        features: List[np.ndarray] = []
        labels: List[np.ndarray] = []
        sample_rates: List[float] = []
        row_buffer = np.empty((0, 8), dtype=np.float64)
        buffer_start = 0
        next_window_start = 0
        rows_seen = 0
        first_timestamp: float | None = None
        last_timestamp: float | None = None

        with zipfile.ZipFile(session["archive_path"]) as archive:
            with archive.open(session["motion_member"]) as motion_stream, archive.open(
                session["label_member"]
            ) as label_stream:
                motion_chunks = pd.read_csv(
                    motion_stream,
                    sep=r"\s+",
                    header=None,
                    usecols=range(7),
                    na_values=["NaN"],
                    chunksize=self.chunk_size,
                )
                label_chunks = pd.read_csv(
                    label_stream,
                    sep=r"\s+",
                    header=None,
                    usecols=[0, 1],
                    chunksize=self.chunk_size,
                )
                for motion_frame, label_frame in zip_longest(
                    motion_chunks, label_chunks
                ):
                    if motion_frame is None or label_frame is None:
                        raise ValueError(
                            f"SHL sensor and label streams differ in length for "
                            f"{session['session_id']}."
                        )
                    motion = motion_frame.to_numpy(dtype=np.float64)
                    label_values = label_frame.to_numpy(dtype=np.float64)
                    if len(motion) != len(label_values):
                        raise ValueError(
                            f"SHL sensor and label chunks differ in length for "
                            f"{session['session_id']}."
                        )
                    if not np.array_equal(
                        np.rint(motion[:, 0]).astype(np.int64),
                        np.rint(label_values[:, 0]).astype(np.int64),
                    ):
                        raise ValueError(
                            f"SHL sensor and label timestamps do not align for "
                            f"{session['session_id']}."
                        )

                    chunk_times = motion[:, 0] / 1000.0
                    if first_timestamp is None and len(chunk_times):
                        first_timestamp = float(chunk_times[0])
                    if len(chunk_times):
                        last_timestamp = float(chunk_times[-1])
                    if len(chunk_times) > 1:
                        median_interval = float(np.median(np.diff(chunk_times)))
                        if median_interval > 0:
                            sample_rates.append(1.0 / median_interval)

                    rows = np.column_stack(
                        (
                            chunk_times,
                            label_values[:, 1],
                            motion[:, 1:7],
                        )
                    )
                    combined = np.concatenate((row_buffer, rows), axis=0)
                    end_index = buffer_start + len(combined)
                    last_start = end_index - WINDOW_SAMPLES
                    starts = (
                        np.arange(
                            next_window_start,
                            last_start + 1,
                            WINDOW_HOP,
                            dtype=int,
                        )
                        if next_window_start <= last_start
                        else np.empty(0, dtype=int)
                    )
                    local_starts = starts - buffer_start
                    valid_starts, mapped_labels = _valid_window_positions(
                        combined, local_starts
                    )
                    if len(valid_starts):
                        windows = np.stack(
                            [
                                combined[start : start + WINDOW_SAMPLES, 2:]
                                for start in valid_starts
                            ]
                        )
                        timestamps.append(combined[valid_starts, 0])
                        features.append(extract_window_features(windows))
                        labels.append(mapped_labels)

                    if len(starts):
                        next_window_start = int(starts[-1] + WINDOW_HOP)
                    keep_from = max(0, next_window_start - buffer_start)
                    row_buffer = combined[keep_from:]
                    buffer_start += keep_from
                    rows_seen += len(rows)

        if sample_rates:
            sampling_rate = float(np.median(sample_rates))
            if not SHL_MIN_SAMPLE_RATE_HZ <= sampling_rate <= SHL_MAX_SAMPLE_RATE_HZ:
                raise ValueError(
                    f"SHL Hand IMU must be approximately 100 Hz; received "
                    f"{sampling_rate:.2f} Hz in {session['session_id']}."
                )
        else:
            sampling_rate = SHL_SAMPLE_RATE_HZ

        mapped_activity_ids = (
            set(np.concatenate(labels).tolist()) if labels else set()
        )
        return {
            **session,
            "recording_start": first_timestamp or 0.0,
            "recording_duration_sec": (
                max(0.0, last_timestamp - first_timestamp)
                if first_timestamp is not None and last_timestamp is not None
                else 0.0
            ),
            "sample_count": rows_seen,
            "sample_rate_hz": sampling_rate,
            "timestamps": (
                np.concatenate(timestamps) if timestamps else np.empty(0)
            ),
            "features": (
                np.vstack(features)
                if features
                else np.empty((0, 54), dtype=np.float32)
            ),
            "labels": np.concatenate(labels) if labels else np.empty(0, dtype=int),
            "window_sec": SHL_WINDOW_SEC,
            "sensor_channels": list(SHL_SENSOR_COLUMNS),
            "activities": {
                TARGET_ACTIVITIES[target]: SHL_RAW_ACTIVITY_NAMES[source]
                for source, target in SHL_ACTIVITY_MAP.items()
                if target in mapped_activity_ids
            },
        }

    def load_sessions(self) -> List[Dict[str, Any]]:
        return [
            self.load_session(session)
            for session in self.available_sessions()
        ]

    def get_activity_preview(
        self,
        session: Dict[str, str],
        activity: str,
        duration_sec: float = 6.0,
    ) -> Dict[str, Any]:
        """Return the first measured contiguous Hand-IMU segment for one mapped class."""
        if activity not in SHL_LABEL_NAMES:
            raise ValueError(
                f"SHL does not include a training label for activity '{activity}'."
            )
        target_raw_label = next(
            source
            for source, target in SHL_ACTIVITY_MAP.items()
            if TARGET_ACTIVITIES[target] == activity
        )
        required_rows = max(2, int(round(duration_sec * SHL_SAMPLE_RATE_HZ)))
        carries = np.empty((0, 8), dtype=np.float64)
        with zipfile.ZipFile(session["archive_path"]) as archive:
            with archive.open(session["motion_member"]) as motion_stream, archive.open(
                session["label_member"]
            ) as label_stream:
                motion_chunks = pd.read_csv(
                    motion_stream,
                    sep=r"\s+",
                    header=None,
                    usecols=range(7),
                    na_values=["NaN"],
                    chunksize=self.chunk_size,
                )
                label_chunks = pd.read_csv(
                    label_stream,
                    sep=r"\s+",
                    header=None,
                    usecols=[0, 1],
                    chunksize=self.chunk_size,
                )
                for motion_frame, label_frame in zip_longest(
                    motion_chunks, label_chunks
                ):
                    if motion_frame is None or label_frame is None:
                        break
                    motion = motion_frame.to_numpy(dtype=np.float64)
                    label_values = label_frame.to_numpy(dtype=np.float64)
                    if len(motion) != len(label_values):
                        raise ValueError(
                            f"SHL sensor and label chunks differ in length for "
                            f"{session['session_id']}."
                        )
                    if not np.array_equal(
                        np.rint(motion[:, 0]).astype(np.int64),
                        np.rint(label_values[:, 0]).astype(np.int64),
                    ):
                        raise ValueError(
                            f"SHL sensor and label timestamps do not align for "
                            f"{session['session_id']}."
                        )
                    rows = np.column_stack(
                        (
                            motion[:, 0] / 1000.0,
                            label_values[:, 1],
                            motion[:, 1:7],
                        )
                    )
                    combined = np.concatenate((carries, rows), axis=0)
                    if len(combined) >= required_rows:
                        candidates = np.flatnonzero(
                            combined[:, 1] == target_raw_label
                        )
                        candidates = candidates[
                            candidates + required_rows <= len(combined)
                        ]
                        for start in candidates:
                            candidate = combined[start : start + required_rows]
                            if (
                                np.all(candidate[:, 1] == target_raw_label)
                                and np.isfinite(candidate[:, 2:]).all()
                                and np.all(
                                    np.diff(candidate[:, 0])
                                    <= MAX_CONTIGUOUS_SAMPLE_GAP_SEC
                                )
                            ):
                                origin = candidate[0, 0]
                                return {
                                    "timestamps": candidate[:, 0] - origin,
                                    **{
                                        name: candidate[:, i + 2]
                                        for i, name in enumerate(SHL_SENSOR_COLUMNS)
                                    },
                                    "sampling_rate_hz": SHL_SAMPLE_RATE_HZ,
                                    "data_source": "SHL measured Hand IMU recording",
                                }
                    carries = combined[-(required_rows - 1) :]
        raise ValueError(
            f"No contiguous {duration_sec:g}s SHL Hand IMU segment for "
            f"{activity} in {session['session_id']}."
        )


class SHLActivityModel:
    """Classifier for the three SHL activities shared with SensorQA."""

    def __init__(self) -> None:
        self.classifier = HistGradientBoostingClassifier(
            max_iter=120,
            learning_rate=0.08,
            max_leaf_nodes=31,
            min_samples_leaf=20,
            l2_regularization=1.0,
            class_weight="balanced",
            random_state=42,
        )
        self.is_trained = False

    def fit(self, features: np.ndarray, labels: np.ndarray) -> "SHLActivityModel":
        features = np.asarray(features, dtype=np.float32)
        labels = np.asarray(labels, dtype=int)
        if features.ndim != 2 or features.shape[1] != 54 or not len(features):
            raise ValueError("SHL training features must be a non-empty 54-column matrix.")
        if not set(np.unique(labels)).issubset(SHL_ACTIVITY_MAP.values()):
            raise ValueError("SHL labels contain activities outside the shared mapping.")
        if len(np.unique(labels)) < 2:
            raise ValueError("SHL training requires at least two mapped activities.")
        self.classifier.fit(features, labels)
        self.is_trained = True
        return self

    def predict(
        self, features: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        if not self.is_trained:
            raise RuntimeError("SHL activity model has not been trained.")
        probabilities = self.classifier.predict_proba(
            np.asarray(features, dtype=np.float32)
        )
        labels = self.classifier.classes_[np.argmax(probabilities, axis=1)]
        return labels.astype(int), probabilities.max(axis=1)

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "wb") as file:
            pickle.dump(
                {
                    "version": SHL_CHECKPOINT_VERSION,
                    "classifier": self.classifier,
                    "is_trained": self.is_trained,
                    "activity_map": SHL_ACTIVITY_MAP,
                    "sensor_location": "hand",
                    "sensor_channels": SHL_SENSOR_COLUMNS,
                    "sample_rate_hz": SHL_SAMPLE_RATE_HZ,
                    "window_sec": SHL_WINDOW_SEC,
                },
                file,
            )

    @classmethod
    def load(cls, path: str) -> "SHLActivityModel":
        with open(path, "rb") as file:
            checkpoint = pickle.load(file)
        if (
            checkpoint.get("version") != SHL_CHECKPOINT_VERSION
            or checkpoint.get("activity_map") != SHL_ACTIVITY_MAP
            or checkpoint.get("sensor_location") != "hand"
        ):
            raise ValueError("SHL model checkpoint metadata is incompatible.")
        model = cls()
        model.classifier = checkpoint["classifier"]
        model.is_trained = bool(checkpoint["is_trained"])
        return model


def evaluate_leave_one_subject_out(
    sessions: Sequence[Dict[str, Any]],
    include_session_predictions: bool = False,
) -> Dict[str, Any]:
    """Evaluate with all days from one participant held out together."""
    from sklearn.metrics import accuracy_score, f1_score

    by_subject: Dict[str, List[Dict[str, Any]]] = {}
    for session in sessions:
        by_subject.setdefault(session["subject"], []).append(session)
    if len(by_subject) < 2:
        raise ValueError("SHL participant-held-out evaluation requires at least two users.")

    all_classes = sorted(SHL_ACTIVITY_MAP.values())
    predictions: List[np.ndarray] = []
    targets: List[np.ndarray] = []
    per_subject: Dict[str, Dict[str, Any]] = {}
    session_predictions: Dict[str, Dict[str, np.ndarray]] = {}
    for held_out_subject, held_out_sessions in sorted(by_subject.items()):
        training_sessions = [
            session
            for subject, subject_sessions in by_subject.items()
            if subject != held_out_subject
            for session in subject_sessions
            if len(session["labels"])
        ]
        test_sessions = [
            session for session in held_out_sessions if len(session["labels"])
        ]
        if not training_sessions or not test_sessions:
            continue
        train_features = np.vstack(
            [session["features"] for session in training_sessions]
        )
        train_labels = np.concatenate(
            [session["labels"] for session in training_sessions]
        )
        if len(np.unique(train_labels)) < 2:
            raise ValueError(
                f"SHL training split without {held_out_subject} has fewer than "
                "two mapped activity classes."
            )
        test_features = np.vstack([session["features"] for session in test_sessions])
        test_labels = np.concatenate([session["labels"] for session in test_sessions])
        model = SHLActivityModel().fit(train_features, train_labels)
        predicted, confidences = model.predict(test_features)
        predictions.append(predicted)
        targets.append(test_labels)
        offset = 0
        for session in test_sessions:
            end = offset + len(session["labels"])
            session_predictions[session["session_id"]] = {
                "predictions": predicted[offset:end],
                "confidences": confidences[offset:end],
            }
            offset = end
        per_subject[held_out_subject] = {
            "windows": len(test_labels),
            "accuracy": float(accuracy_score(test_labels, predicted)),
            "macro_f1": float(
                f1_score(
                    test_labels,
                    predicted,
                    labels=all_classes,
                    average="macro",
                    zero_division=0,
                )
            ),
        }
    if not targets:
        raise ValueError("No SHL participant windows were available for evaluation.")
    y_true = np.concatenate(targets)
    y_pred = np.concatenate(predictions)
    metrics = {
        "dataset": "University of Sussex-Huawei Locomotion (SHL) Dataset preview",
        "evaluation": "leave-one-participant-out; pooled out-of-fold predictions",
        "subjects": list(per_subject),
        "subject_count": len(per_subject),
        "window_seconds": SHL_WINDOW_SEC,
        "window_hop_seconds": SHL_WINDOW_HOP_SEC,
        "timeline_bin_seconds": SHL_TIMELINE_STEP_SEC,
        "sample_rate_hz": SHL_SAMPLE_RATE_HZ,
        "sensor_location": "hand",
        "sensor_channels": list(SHL_SENSOR_COLUMNS),
        "activities": {
            TARGET_ACTIVITIES[index]: SHL_LABEL_NAMES[TARGET_ACTIVITIES[index]]
            for index in all_classes
        },
        "class_distribution": {
            TARGET_ACTIVITIES[index]: int(np.sum(y_true == index))
            for index in all_classes
        },
        "windows": len(y_true),
        "pooled_accuracy": float(accuracy_score(y_true, y_pred)),
        "pooled_macro_f1": float(
            f1_score(
                y_true,
                y_pred,
                labels=all_classes,
                average="macro",
                zero_division=0,
            )
        ),
        "per_subject": per_subject,
        "excluded_labels": {
            str(raw_label): name
            for raw_label, name in SHL_RAW_ACTIVITY_NAMES.items()
            if raw_label not in SHL_ACTIVITY_MAP
        },
    }
    if include_session_predictions:
        return {
            "metrics": metrics,
            "session_predictions": session_predictions,
        }
    return metrics
