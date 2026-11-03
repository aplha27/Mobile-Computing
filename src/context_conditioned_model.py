"""Activity classification using ExtraSensory features and observed annotations."""

import os
import pickle
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score

from src.dataset_loader import EXTRASENSORY_CONTEXT_LABELS, EXTRASENSORY_LOCATION_LABELS


ANNOTATION_COLUMNS = tuple(
    (kind, label)
    for kind, labels in (
        ("context", EXTRASENSORY_CONTEXT_LABELS),
        ("location", EXTRASENSORY_LOCATION_LABELS),
    )
    for label in labels
)


def encode_annotation_features(
    context_labels: Dict[str, np.ndarray],
    location_labels: Dict[str, np.ndarray],
    row_count: int,
) -> np.ndarray:
    """Encode observed values and observation masks so missing is not negative."""
    encoded: List[np.ndarray] = []
    for kind, label in ANNOTATION_COLUMNS:
        source = context_labels if kind == "context" else location_labels
        values = np.asarray(
            source.get(label, np.full(row_count, np.nan)), dtype=float
        )
        if values.shape != (row_count,):
            raise ValueError(f"{kind} label '{label}' must have {row_count} rows.")
        observed = np.isfinite(values)
        encoded.extend((np.where(observed, values, 0.0), observed.astype(float)))
    return np.column_stack(encoded)


def _session_arrays(sessions: Iterable[Dict[str, Any]]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    features: List[np.ndarray] = []
    targets: List[np.ndarray] = []
    annotations: List[np.ndarray] = []
    for session in sessions:
        valid = np.asarray(session["valid_mask"], dtype=bool)
        if not np.any(valid):
            continue
        raw_features = np.asarray(session["raw_features"], dtype=float)[valid]
        features.append(np.nan_to_num(raw_features, nan=0.0, posinf=0.0, neginf=0.0))
        targets.append(np.asarray(session["labels"], dtype=int)[valid])
        annotations.append(
            encode_annotation_features(
                session["context_labels"], session["location_labels"], len(valid)
            )[valid]
        )
    if not features:
        raise ValueError("No activity-labeled rows were found in the supplied sessions.")
    return np.vstack(features), np.concatenate(targets), np.vstack(annotations)


def _balance_rows(
    features: np.ndarray,
    targets: np.ndarray,
    annotations: np.ndarray,
    maximum_per_class: int,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    selected = []
    for target in np.unique(targets):
        indexes = np.flatnonzero(targets == target)
        if maximum_per_class > 0 and len(indexes) > maximum_per_class:
            indexes = rng.choice(indexes, size=maximum_per_class, replace=False)
        selected.extend(indexes.tolist())
    indexes = np.asarray(selected, dtype=int)
    rng.shuffle(indexes)
    return (
        np.hstack((features[indexes], annotations[indexes])),
        targets[indexes],
    )


def _decode_annotation_features(
    encoded: np.ndarray,
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    context: Dict[str, np.ndarray] = {}
    location: Dict[str, np.ndarray] = {}
    for index, (kind, label) in enumerate(ANNOTATION_COLUMNS):
        values = encoded[:, index * 2]
        observed = encoded[:, index * 2 + 1].astype(bool)
        destination = context if kind == "context" else location
        destination[label] = np.where(observed, values, np.nan)
    return context, location


class ContextConditionedActivityModel:
    """A distinct classifier that can use observed context/place annotations."""

    def __init__(self) -> None:
        self.classifier = HistGradientBoostingClassifier(
            max_iter=120,
            learning_rate=0.06,
            max_leaf_nodes=31,
            min_samples_leaf=20,
            l2_regularization=1.0,
            class_weight="balanced",
            random_state=42,
        )
        self.feature_count = 0
        self.is_trained = False

    def fit(
        self,
        sessions: Sequence[Dict[str, Any]],
        maximum_per_class: int = 5000,
    ) -> "ContextConditionedActivityModel":
        features, targets, annotations = _session_arrays(sessions)
        train_x, train_y = _balance_rows(
            features, targets, annotations, maximum_per_class, seed=42
        )
        self.feature_count = features.shape[1]
        self.classifier.fit(train_x, train_y)
        self.is_trained = True
        return self

    def predict(
        self,
        raw_features: np.ndarray,
        context_labels: Dict[str, np.ndarray],
        location_labels: Dict[str, np.ndarray],
    ) -> Tuple[np.ndarray, np.ndarray]:
        if not self.is_trained:
            raise RuntimeError("Context-conditioned model has not been trained.")
        raw_features = np.asarray(raw_features, dtype=float)
        if raw_features.ndim != 2 or raw_features.shape[1] != self.feature_count:
            raise ValueError(
                f"Expected a 2D feature matrix with {self.feature_count} columns."
            )
        raw_features = np.nan_to_num(raw_features, nan=0.0, posinf=0.0, neginf=0.0)
        annotations = encode_annotation_features(
            context_labels, location_labels, len(raw_features)
        )
        probabilities = self.classifier.predict_proba(
            np.hstack((raw_features, annotations))
        )
        aligned = np.zeros((len(raw_features), 7), dtype=float)
        for column, class_id in enumerate(self.classifier.classes_):
            if 0 <= class_id < aligned.shape[1]:
                aligned[:, class_id] = probabilities[:, column]
        return np.argmax(aligned, axis=1), np.max(aligned, axis=1)

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "wb") as file:
            pickle.dump(
                {
                    "classifier": self.classifier,
                    "feature_count": self.feature_count,
                    "is_trained": self.is_trained,
                },
                file,
            )

    @classmethod
    def load(cls, path: str) -> "ContextConditionedActivityModel":
        with open(path, "rb") as file:
            checkpoint = pickle.load(file)
        model = cls()
        model.classifier = checkpoint["classifier"]
        model.feature_count = int(checkpoint["feature_count"])
        model.is_trained = bool(checkpoint["is_trained"])
        return model


def evaluate_context_conditioned_model(
    training_sessions: Sequence[Dict[str, Any]],
    test_sessions: Sequence[Dict[str, Any]],
) -> Tuple[ContextConditionedActivityModel, Dict[str, Any]]:
    """Evaluate on unseen users; reserve training users for validation."""
    ordered = sorted(training_sessions, key=lambda session: session["uuid"])
    if len(ordered) < 2:
        raise ValueError("At least two training users are required for held-out validation.")
    validation_count = max(1, int(round(len(ordered) * 0.15)))
    validation_sessions = ordered[:validation_count]
    fit_sessions = ordered[validation_count:]

    validation_model = ContextConditionedActivityModel().fit(fit_sessions)
    val_x, val_y, val_annotations = _session_arrays(validation_sessions)
    val_context, val_location = _decode_annotation_features(val_annotations)
    val_pred, _ = validation_model.predict(val_x, val_context, val_location)

    final_model = ContextConditionedActivityModel().fit(training_sessions)
    test_x, test_y, test_annotations = _session_arrays(test_sessions)
    test_context, test_location = _decode_annotation_features(test_annotations)
    test_pred, _ = final_model.predict(test_x, test_context, test_location)

    context_indexes = [
        index * 2
        for index, (kind, _) in enumerate(ANNOTATION_COLUMNS)
        if kind == "context"
    ]
    location_indexes = [
        index * 2
        for index, (kind, _) in enumerate(ANNOTATION_COLUMNS)
        if kind == "location"
    ]
    return final_model, {
        "validation_macro_f1": float(
            f1_score(val_y, val_pred, average="macro", zero_division=0)
        ),
        "test_macro_f1": float(
            f1_score(test_y, test_pred, average="macro", zero_division=0)
        ),
        "test_weighted_f1": float(
            f1_score(test_y, test_pred, average="weighted", zero_division=0)
        ),
        "test_accuracy": float(accuracy_score(test_y, test_pred)),
        "test_users": len(test_sessions),
        "test_rows": len(test_y),
        "test_context_observation_fraction": float(
            np.mean(test_annotations[:, [index + 1 for index in context_indexes]])
        ),
        "test_place_observation_fraction": float(
            np.mean(test_annotations[:, [index + 1 for index in location_indexes]])
        ),
    }
