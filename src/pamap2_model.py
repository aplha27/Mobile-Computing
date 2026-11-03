"""PAMAP2 raw-signal windowing, activity mapping, and classifier."""

import os
import pickle
import zipfile
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

from src.dataset_loader import TARGET_ACTIVITIES


PAMAP2_ARCHIVE = "pamap2+physical+activity+monitoring.zip"
PAMAP2_INNER_ARCHIVE = "PAMAP2_Dataset.zip"
PAMAP2_PROTOCOL_PREFIX = "PAMAP2_Dataset/Protocol/"
PAMAP2_ACTIVITY_MAP = {
    1: 0,  # lying
    2: 1,  # sitting
    3: 2,  # standing
    4: 4,  # walking
    5: 5,  # running
    6: 6,  # cycling
}
PAMAP2_LABEL_NAMES = {
    "lying_down": "lying",
    "sitting": "sitting",
    "standing_in_place": "standing",
    "walking": "walking",
    "running": "running",
    "bicycling": "cycling",
}
PAMAP2_SOURCE_COLUMNS = [0, 1, 25, 26, 27, 28, 29, 30]
PAMAP2_SENSOR_COLUMNS = ["Acc X", "Acc Y", "Acc Z", "Gyro X", "Gyro Y", "Gyro Z"]
WINDOW_SEC = 5.0
WINDOW_SAMPLES = 500
WINDOW_HOP = 250
SAMPLE_RATE_HZ = 100.0
TIMELINE_STEP_SEC = WINDOW_HOP / SAMPLE_RATE_HZ
MAX_CONTIGUOUS_SAMPLE_GAP_SEC = 0.02
MIN_SUPPORTED_SAMPLE_RATE_HZ = 90.0
MAX_SUPPORTED_SAMPLE_RATE_HZ = 110.0


def extract_window_features(windows: np.ndarray) -> np.ndarray:
    """Extract robust time-domain statistics from (windows, samples, channels)."""
    windows = np.asarray(windows, dtype=np.float32)
    if windows.ndim != 3 or windows.shape[2] != 6:
        raise ValueError("Expected windows shaped (n, samples, 6).")

    channel_features = [
        np.mean(windows, axis=1),
        np.std(windows, axis=1),
        np.min(windows, axis=1),
        np.max(windows, axis=1),
        np.sqrt(np.mean(np.square(windows), axis=1)),
        np.percentile(windows, 10, axis=1),
        np.percentile(windows, 90, axis=1),
        np.mean(np.abs(np.diff(windows, axis=1)), axis=1),
    ]
    features = np.concatenate(channel_features, axis=1)
    for group in (windows[:, :, :3], windows[:, :, 3:]):
        magnitude = np.linalg.norm(group, axis=2)
        features = np.column_stack(
            (
                features,
                magnitude.mean(axis=1),
                magnitude.std(axis=1),
                np.sqrt(np.mean(np.square(magnitude), axis=1)),
            )
        )
    return np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)


def make_labeled_windows(
    rows: np.ndarray,
    window_samples: int = WINDOW_SAMPLES,
    hop_samples: int = WINDOW_HOP,
) -> Dict[str, Any]:
    """Create non-overlapping-label 5 s windows from PAMAP2 100 Hz raw rows."""
    rows = np.asarray(rows, dtype=float)
    if rows.ndim != 2 or rows.shape[1] != 8:
        raise ValueError("PAMAP2 rows must contain timestamp, activity, and 6 channels.")
    if len(rows) < window_samples:
        return {
            "timestamps": np.empty(0),
            "features": np.empty((0, 54), dtype=np.float32),
            "labels": np.empty(0, dtype=int),
        }

    timestamps = rows[:, 0]
    activity_ids = rows[:, 1].astype(int)
    sensors = rows[:, 2:]
    candidates = np.arange(0, len(rows) - window_samples + 1, hop_samples)
    mapped_labels = np.asarray(
        [PAMAP2_ACTIVITY_MAP.get(int(activity_ids[start]), -1) for start in candidates],
        dtype=int,
    )
    keep = mapped_labels >= 0
    label_changes = np.concatenate(
        ([0], np.cumsum(activity_ids[1:] != activity_ids[:-1]))
    )
    keep &= (
        label_changes[candidates + window_samples - 1]
        - label_changes[candidates]
    ) == 0

    invalid_sensor_rows = ~np.all(np.isfinite(sensors), axis=1)
    invalid_sensor_prefix = np.concatenate(
        ([0], np.cumsum(invalid_sensor_rows))
    )
    keep &= (
        invalid_sensor_prefix[candidates + window_samples]
        - invalid_sensor_prefix[candidates]
    ) == 0

    sample_gaps = np.diff(timestamps)
    invalid_gap_prefix = np.concatenate(
        (
            [0],
            np.cumsum(
                (sample_gaps <= 0)
                | (sample_gaps > MAX_CONTIGUOUS_SAMPLE_GAP_SEC)
            ),
        )
    )
    keep &= (
        invalid_gap_prefix[candidates + window_samples - 1]
        - invalid_gap_prefix[candidates]
    ) == 0

    starts = candidates[keep]
    labels = mapped_labels[keep]
    if not len(starts):
        return {
            "timestamps": np.empty(0),
            "features": np.empty((0, 54), dtype=np.float32),
            "labels": np.empty(0, dtype=int),
        }
    windows = np.stack(
        [sensors[start : start + window_samples] for start in starts]
    )
    return {
        "timestamps": timestamps[starts],
        "features": extract_window_features(windows),
        "labels": labels,
    }


def make_unlabeled_windows(
    rows: np.ndarray,
    window_samples: int = WINDOW_SAMPLES,
    hop_samples: int = WINDOW_HOP,
) -> Dict[str, np.ndarray]:
    """Window timestamp plus six raw channels for inference without labels."""
    rows = np.asarray(rows, dtype=float)
    if rows.ndim != 2 or rows.shape[1] != 7:
        raise ValueError("Raw inference rows must contain timestamp and 6 channels.")
    if len(rows) < window_samples:
        return {
            "timestamps": np.empty(0),
            "features": np.empty((0, 54), dtype=np.float32),
        }

    timestamps = rows[:, 0]
    sensors = rows[:, 1:]
    sample_rate = 1.0 / float(np.median(np.diff(timestamps)))
    if not MIN_SUPPORTED_SAMPLE_RATE_HZ <= sample_rate <= MAX_SUPPORTED_SAMPLE_RATE_HZ:
        raise ValueError(
            "PAMAP2 inference requires approximately 100 Hz input "
            f"(received {sample_rate:.2f} Hz)."
        )
    candidates = np.arange(0, len(rows) - window_samples + 1, hop_samples)
    keep = np.ones(len(candidates), dtype=bool)
    invalid_sensor_prefix = np.concatenate(
        ([0], np.cumsum(~np.all(np.isfinite(sensors), axis=1)))
    )
    keep &= (
        invalid_sensor_prefix[candidates + window_samples]
        - invalid_sensor_prefix[candidates]
    ) == 0
    invalid_gap_prefix = np.concatenate(
        (
            [0],
            np.cumsum(
                np.diff(timestamps) > MAX_CONTIGUOUS_SAMPLE_GAP_SEC
            ),
        )
    )
    keep &= (
        invalid_gap_prefix[candidates + window_samples - 1]
        - invalid_gap_prefix[candidates]
    ) == 0

    starts = candidates[keep]
    if not len(starts):
        return {
            "timestamps": np.empty(0),
            "features": np.empty((0, 54), dtype=np.float32),
        }
    windows = np.stack(
        [sensors[start : start + window_samples] for start in starts]
    )
    return {
        "timestamps": timestamps[starts],
        "features": extract_window_features(windows),
    }


class PAMAP2Loader:
    """Reads the nested official protocol archive without extracting it to disk."""

    def __init__(self, archive_path: str = PAMAP2_ARCHIVE):
        self.archive_path = archive_path

    def _open_protocol(self):
        if not os.path.isfile(self.archive_path):
            raise FileNotFoundError(f"PAMAP2 archive not found: {self.archive_path}")
        return zipfile.ZipFile(self.archive_path)

    def available_subjects(self) -> List[str]:
        with self._open_protocol() as outer:
            inner_name = self._find_inner_archive(outer)
            with outer.open(inner_name) as nested_stream:
                with zipfile.ZipFile(nested_stream) as inner:
                    return sorted(
                        name[len(PAMAP2_PROTOCOL_PREFIX) : -4]
                        for name in inner.namelist()
                        if name.startswith(PAMAP2_PROTOCOL_PREFIX)
                        and name.endswith(".dat")
                    )

    @staticmethod
    def _find_inner_archive(outer: zipfile.ZipFile) -> str:
        candidates = [
            name for name in outer.namelist()
            if name.lower().endswith(PAMAP2_INNER_ARCHIVE.lower())
        ]
        if len(candidates) != 1:
            raise ValueError(
                f"Expected one nested {PAMAP2_INNER_ARCHIVE}; found {len(candidates)}."
            )
        return candidates[0]

    def load_subject(self, subject: str) -> Dict[str, Any]:
        if not subject.startswith("subject") or not subject[7:].isdigit():
            raise ValueError("Subject must use the PAMAP2 subjectNNN form.")
        member = f"{PAMAP2_PROTOCOL_PREFIX}{subject}.dat"
        with self._open_protocol() as outer:
            inner_name = self._find_inner_archive(outer)
            with outer.open(inner_name) as nested_stream:
                with zipfile.ZipFile(nested_stream) as inner:
                    if member not in inner.namelist():
                        raise FileNotFoundError(f"PAMAP2 protocol file not found: {member}")
                    with inner.open(member) as stream:
                        frame = pd.read_csv(
                            stream,
                            sep=r"\s+",
                            header=None,
                            usecols=PAMAP2_SOURCE_COLUMNS,
                            na_values=["NaN"],
                        )
        rows = frame.to_numpy(dtype=float)
        rows = rows[np.isfinite(rows[:, 0]) & np.isfinite(rows[:, 1])]
        windows = make_labeled_windows(rows)
        return {
            "subject": subject,
            "session_id": f"pamap2-{subject}",
            "recording_start": float(rows[0, 0]) if len(rows) else 0.0,
            "recording_duration_sec": (
                float(rows[-1, 0] - rows[0, 0]) if len(rows) else 0.0
            ),
            "timestamps": windows["timestamps"],
            "features": windows["features"],
            "labels": windows["labels"],
            "window_sec": WINDOW_SEC,
            "sample_rate_hz": SAMPLE_RATE_HZ,
            "sensor_channels": list(PAMAP2_SENSOR_COLUMNS),
            "activities": {
                TARGET_ACTIVITIES[index]: PAMAP2_LABEL_NAMES[TARGET_ACTIVITIES[index]]
                for index in sorted(set(windows["labels"].tolist()))
            },
        }

    def load_protocol_subjects(self) -> List[Dict[str, Any]]:
        return [
            self.load_subject(subject)
            for subject in self.available_subjects()
            if int(subject[7:]) <= 108
        ]

    def get_activity_preview(
        self, subject: str, activity: str, duration_sec: float = 6.0
    ) -> Dict[str, Any]:
        if activity not in PAMAP2_LABEL_NAMES:
            raise ValueError(f"Activity '{activity}' is not included in PAMAP2 mapping.")
        activity_id = next(
            key for key, target in PAMAP2_ACTIVITY_MAP.items()
            if TARGET_ACTIVITIES[target] == activity
        )
        required_rows = max(2, int(round(duration_sec * SAMPLE_RATE_HZ)))
        member = f"{PAMAP2_PROTOCOL_PREFIX}{subject}.dat"
        with self._open_protocol() as outer:
            inner_name = self._find_inner_archive(outer)
            with outer.open(inner_name) as nested_stream:
                with zipfile.ZipFile(nested_stream) as inner:
                    if member not in inner.namelist():
                        raise FileNotFoundError(f"PAMAP2 protocol file not found: {member}")
                    with inner.open(member) as stream:
                        for chunk in pd.read_csv(
                            stream,
                            sep=r"\s+",
                            header=None,
                            usecols=PAMAP2_SOURCE_COLUMNS,
                            na_values=["NaN"],
                            chunksize=100_000,
                        ):
                            rows = chunk.to_numpy(dtype=float)
                            labels = rows[:, 1]
                            matches = np.flatnonzero(labels == activity_id)
                            for start in matches:
                                end = int(start) + required_rows
                                if end > len(rows):
                                    continue
                                candidate = rows[start:end]
                                if (
                                    np.all(candidate[:, 1] == activity_id)
                                    and np.isfinite(candidate[:, [0, 2, 3, 4, 5, 6, 7]]).all()
                                    and np.all(
                                        np.diff(candidate[:, 0])
                                        <= MAX_CONTIGUOUS_SAMPLE_GAP_SEC
                                    )
                                ):
                                    relative = candidate[:, 0] - candidate[0, 0]
                                    return {
                                        "timestamps": relative,
                                        **{
                                            name: candidate[:, i + 2]
                                            for i, name in enumerate(PAMAP2_SENSOR_COLUMNS)
                                        },
                                        "sampling_rate_hz": SAMPLE_RATE_HZ,
                                        "data_source": "PAMAP2 raw protocol recording",
                                    }
        raise ValueError(f"No contiguous {duration_sec:g}s {activity} segment found in {subject}.")


class PAMAP2ActivityModel:
    """Classifier over 5-second features extracted from PAMAP2 raw IMU windows."""

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

    def fit(self, features: np.ndarray, labels: np.ndarray) -> "PAMAP2ActivityModel":
        features = np.asarray(features, dtype=np.float32)
        labels = np.asarray(labels, dtype=int)
        if features.ndim != 2 or not len(features):
            raise ValueError("PAMAP2 training features must be a non-empty 2D matrix.")
        if not set(np.unique(labels)).issubset(PAMAP2_ACTIVITY_MAP.values()):
            raise ValueError("Training labels contain activities outside the PAMAP2 mapping.")
        self.classifier.fit(features, labels)
        self.is_trained = True
        return self

    def predict(
        self, features: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        if not self.is_trained:
            raise RuntimeError("PAMAP2 activity model has not been trained.")
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
                    "classifier": self.classifier,
                    "is_trained": self.is_trained,
                    "window_sec": WINDOW_SEC,
                    "sample_rate_hz": SAMPLE_RATE_HZ,
                    "channels": PAMAP2_SENSOR_COLUMNS,
                    "activity_map": PAMAP2_ACTIVITY_MAP,
                },
                file,
            )

    @classmethod
    def load(cls, path: str) -> "PAMAP2ActivityModel":
        with open(path, "rb") as file:
            checkpoint = pickle.load(file)
        if checkpoint.get("activity_map") != PAMAP2_ACTIVITY_MAP:
            raise ValueError("PAMAP2 model checkpoint activity mapping is incompatible.")
        model = cls()
        model.classifier = checkpoint["classifier"]
        model.is_trained = bool(checkpoint["is_trained"])
        return model


def evaluate_leave_one_subject_out(
    sessions: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    """Generate out-of-fold predictions with whole participants held out."""
    from sklearn.metrics import accuracy_score, f1_score

    predictions: List[np.ndarray] = []
    targets: List[np.ndarray] = []
    per_subject = {}
    for held_out in sessions:
        training = [item for item in sessions if item["subject"] != held_out["subject"]]
        train_x = np.vstack([item["features"] for item in training if len(item["labels"])])
        train_y = np.concatenate([item["labels"] for item in training if len(item["labels"])])
        if not len(held_out["labels"]):
            continue
        model = PAMAP2ActivityModel().fit(train_x, train_y)
        predicted, _ = model.predict(held_out["features"])
        predictions.append(predicted)
        targets.append(held_out["labels"])
        per_subject[held_out["subject"]] = {
            "windows": len(held_out["labels"]),
            "accuracy": float(accuracy_score(held_out["labels"], predicted)),
            "macro_f1": float(
                f1_score(
                    held_out["labels"],
                    predicted,
                    labels=sorted(PAMAP2_ACTIVITY_MAP.values()),
                    average="macro",
                    zero_division=0,
                )
            ),
        }

    if not predictions:
        raise ValueError("No PAMAP2 participant windows were available for evaluation.")
    y_true = np.concatenate(targets)
    y_pred = np.concatenate(predictions)
    classes = sorted(PAMAP2_ACTIVITY_MAP.values())
    return {
        "dataset": "PAMAP2 Protocol",
        "evaluation": "leave-one-subject-out; pooled out-of-fold predictions",
        "subjects": list(per_subject),
        "subject_count": len(per_subject),
        "window_seconds": WINDOW_SEC,
        "window_hop_seconds": WINDOW_HOP / SAMPLE_RATE_HZ,
        "timeline_bin_seconds": TIMELINE_STEP_SEC,
        "sample_rate_hz": SAMPLE_RATE_HZ,
        "sensor_location": "chest IMU",
        "sensor_channels": list(PAMAP2_SENSOR_COLUMNS),
        "activities": {
            TARGET_ACTIVITIES[index]: PAMAP2_LABEL_NAMES[TARGET_ACTIVITIES[index]]
            for index in classes
        },
        "class_distribution": {
            TARGET_ACTIVITIES[index]: int(np.sum(y_true == index))
            for index in classes
        },
        "windows": len(y_true),
        "pooled_accuracy": float(accuracy_score(y_true, y_pred)),
        "pooled_macro_f1": float(
            f1_score(
                y_true, y_pred, labels=classes, average="macro", zero_division=0
            )
        ),
        "per_subject": per_subject,
        "excluded_subject": "subject109 (protocol file contains only activity 24: rope jumping)",
    }
