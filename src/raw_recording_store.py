"""In-memory import and preview support for real six-axis sensor CSV files."""

from collections import OrderedDict
from typing import Dict, Any
import io
import uuid

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = (
    "timestamp",
    "acc_x",
    "acc_y",
    "acc_z",
    "gyro_x",
    "gyro_y",
    "gyro_z",
)
CHANNEL_COLUMNS = {
    "acc_x": "Acc X",
    "acc_y": "Acc Y",
    "acc_z": "Acc Z",
    "gyro_x": "Gyro X",
    "gyro_y": "Gyro Y",
    "gyro_z": "Gyro Z",
}
MAX_RECORDING_ROWS = 200_000
MAX_CACHED_RECORDINGS = 3


class RawRecordingStore:
    def __init__(self) -> None:
        self._recordings: OrderedDict[str, Dict[str, Any]] = OrderedDict()

    def add_csv(self, content: bytes) -> Dict[str, Any]:
        try:
            frame = pd.read_csv(io.BytesIO(content))
        except (pd.errors.EmptyDataError, pd.errors.ParserError, UnicodeDecodeError) as error:
            raise ValueError("Upload must be a valid UTF-8 CSV file.") from error

        missing = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
        if missing:
            raise ValueError(
                "CSV is missing required columns: " + ", ".join(missing)
            )
        if len(frame) < 2:
            raise ValueError("CSV must contain at least two sensor samples.")
        if len(frame) > MAX_RECORDING_ROWS:
            raise ValueError(
                f"CSV exceeds the {MAX_RECORDING_ROWS} sample limit."
            )

        numeric = frame.loc[:, REQUIRED_COLUMNS].apply(
            pd.to_numeric, errors="coerce"
        )
        values = numeric.to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValueError("Required timestamp and sensor values must be finite numbers.")

        source_timestamps = values[:, 0]
        differences = np.diff(source_timestamps)
        if np.any(differences <= 0):
            raise ValueError("Timestamps must be strictly increasing in seconds.")

        timestamps = source_timestamps - source_timestamps[0]
        channels = {
            CHANNEL_COLUMNS[column]: values[:, index + 1]
            for index, column in enumerate(REQUIRED_COLUMNS[1:])
        }
        duration = float(timestamps[-1])
        sample_rate = float(1.0 / np.median(differences))
        recording_id = f"raw-{uuid.uuid4().hex}"
        self._recordings[recording_id] = {
            "timestamps": timestamps,
            "channels": channels,
            "duration_sec": duration,
            "sample_rate_hz": sample_rate,
            "sample_count": len(timestamps),
        }
        while len(self._recordings) > MAX_CACHED_RECORDINGS:
            self._recordings.popitem(last=False)

        return {
            "recording_id": recording_id,
            "sample_count": len(timestamps),
            "duration_sec": duration,
            "sample_rate_hz": sample_rate,
            "data_source": "user-provided raw sensor recording",
            "classification": "not performed; raw model training requires labeled recordings",
        }

    def get_signals(self, recording_id: str, limit: int = 1500) -> Dict[str, Any]:
        recording = self._recordings.get(recording_id)
        if recording is None:
            raise KeyError(recording_id)

        limit = min(max(int(limit), 2), 5000)
        timestamps = recording["timestamps"]
        if len(timestamps) > limit:
            indexes = np.linspace(0, len(timestamps) - 1, limit, dtype=int)
        else:
            indexes = np.arange(len(timestamps))

        return {
            "recording_id": recording_id,
            "data_source": "user-provided raw sensor recording",
            "sampling_rate_hz": round(recording["sample_rate_hz"], 3),
            "duration_sec": round(recording["duration_sec"], 3),
            "timestamps": timestamps[indexes].tolist(),
            **{
                key.lower().replace(" ", "_"): values[indexes].tolist()
                for key, values in recording["channels"].items()
            },
        }

    def get_activity_windows(self, recording_id: str) -> Dict[str, np.ndarray]:
        recording = self._recordings.get(recording_id)
        if recording is None:
            raise KeyError(recording_id)
        from src.pamap2_model import make_unlabeled_windows

        rows = np.column_stack(
            (
                recording["timestamps"],
                *[
                    recording["channels"][CHANNEL_COLUMNS[column]]
                    for column in REQUIRED_COLUMNS[1:]
                ],
            )
        )
        windows = make_unlabeled_windows(rows)
        windows["recording_duration_sec"] = recording["duration_sec"]
        return windows

    def contains(self, recording_id: str) -> bool:
        return recording_id in self._recordings
