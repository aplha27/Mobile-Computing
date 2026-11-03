"""
dataset_loader.py
=================
Robust data ingestion and preprocessing pipeline for the ExtraSensory dataset.
Loads its existing activity features, timestamps, context labels, and coarse place
labels for the 7 target activities:
  1. Lying down
  2. Sitting
  3. Standing in place
  4. Standing and moving
  5. Walking
  6. Running
  7. Bicycling

Also includes an illustrative 25 Hz signal synthesizer. Supports:
  - Official 5-fold cross validation splits (cv5Folds.zip)
  - Extra user archives (archive.zip)
  - Multi-user cross-subject generalization
"""

import os
import io
import gzip
import zipfile
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

TARGET_ACTIVITIES = [
    "lying_down",
    "sitting",
    "standing_in_place",
    "standing_and_moving",
    "walking",
    "running",
    "bicycling"
]

ACTIVITY_DISPLAY_NAMES = {
    "lying_down": "Lying down",
    "sitting": "Sitting",
    "standing_in_place": "Standing in place",
    "standing_and_moving": "Standing and moving",
    "walking": "Walking",
    "running": "Running",
    "bicycling": "Bicycling"
}

ACTIVITY_MODALITIES = {
    "lying_down": ["accelerometer", "gyroscope"],
    "sitting": ["accelerometer", "gyroscope"],
    "standing_in_place": ["accelerometer", "gyroscope"],
    "standing_and_moving": ["accelerometer", "gyroscope"],
    "walking": ["accelerometer", "gyroscope"],
    "running": ["accelerometer", "gyroscope"],
    "bicycling": ["accelerometer", "gyroscope"]
}

DEFAULT_CHANNELS = ["Acc X", "Acc Y", "Acc Z", "Gyro X", "Gyro Y", "Gyro Z"]

EXTRASENSORY_CONTEXT_LABELS = {
    "indoors": "label:OR_indoors",
    "outdoors": "label:OR_outside",
    "in_a_car": "label:IN_A_CAR",
    "on_a_bus": "label:ON_A_BUS",
    "driving": "label:DRIVE_-_I_M_THE_DRIVER",
    "passenger": "label:DRIVE_-_I_M_A_PASSENGER",
    "with_friends": "label:WITH_FRIENDS",
    "with_coworkers": "label:WITH_CO-WORKERS",
    "in_a_meeting": "label:IN_A_MEETING",
    "in_class": "label:IN_CLASS",
}

EXTRASENSORY_LOCATION_LABELS = {
    "home": "label:LOC_home",
    "work": "label:LOC_main_workplace",
    "restaurant": "label:FIX_restaurant",
    "gym": "label:AT_THE_GYM",
    "beach": "label:LOC_beach",
    "school": "label:AT_SCHOOL",
}

EXTRASENSORY_LOCATION_FEATURES = [
    "location:num_valid_updates",
    "location:log_latitude_range",
    "location:log_longitude_range",
    "location:min_altitude",
    "location:max_altitude",
    "location:min_speed",
    "location:max_speed",
    "location:best_horizontal_accuracy",
    "location:best_vertical_accuracy",
    "location:diameter",
    "location:log_diameter",
    "location_quick_features:std_lat",
    "location_quick_features:std_long",
    "location_quick_features:lat_change",
    "location_quick_features:long_change",
    "location_quick_features:mean_abs_lat_deriv",
    "location_quick_features:mean_abs_long_deriv",
]


class ExtraSensoryLoader:
    def __init__(
        self,
        zip_path: str = "ExtraSensory.per_uuid_features_labels.zip",
        cv_zip_path: str = "cv5Folds.zip",
        archive_zip_path: str = "archive.zip"
    ):
        self.zip_path = zip_path
        self.cv_zip_path = cv_zip_path
        self.archive_zip_path = archive_zip_path
        self.feature_columns: List[str] = []
        self.scaler = StandardScaler()
        self.is_fitted = False

    def get_available_uuids(self) -> List[str]:
        """Returns list of UUID filenames in the primary dataset."""
        if not os.path.exists(self.zip_path):
            raise FileNotFoundError(f"ExtraSensory archive not found at: {self.zip_path}")
        with zipfile.ZipFile(self.zip_path) as z:
            return sorted([n for n in z.namelist() if n.endswith('.csv.gz')])

    def _extract_7_target_labels(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Maps multi-label ExtraSensory columns to 7 mutually exclusive target activities.
        Hierarchy: Active dynamic locomotion has highest priority over static postures.
        """
        lying = (df.get('label:LYING_DOWN', 0) == 1.0).astype(int)
        sitting = (df.get('label:SITTING', 0) == 1.0).astype(int)
        bicycling = (df.get('label:BICYCLING', 0) == 1.0).astype(int)
        running = (df.get('label:FIX_running', 0) == 1.0).astype(int)
        walking = (df.get('label:FIX_walking', 0) == 1.0).astype(int)
        standing = (df.get('label:OR_standing', 0) == 1.0).astype(int)
        strolling = (df.get('label:STROLLING', 0) == 1.0).astype(int)

        standing_moving = ((standing == 1) & (strolling == 1)) | ((standing == 0) & (strolling == 1))
        standing_place = (standing == 1) & (strolling == 0) & (walking == 0) & (running == 0)

        n = len(df)
        labels = np.full(n, -1, dtype=int)

        # Priority mapping: locomotion first, then postures
        labels[running == 1] = 5          # running
        labels[(bicycling == 1) & (labels == -1)] = 6  # bicycling
        labels[(walking == 1) & (labels == -1)] = 4    # walking
        labels[(standing_moving == 1) & (labels == -1)] = 3  # standing_and_moving
        labels[(standing_place == 1) & (labels == -1)] = 2   # standing_in_place
        labels[(sitting == 1) & (labels == -1)] = 1          # sitting
        labels[(lying == 1) & (labels == -1)] = 0            # lying_down

        valid_mask = labels >= 0
        return labels, valid_mask

    def _select_sensor_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Selects full suite of 6-axis accelerometer & gyroscope kinematic features
        (statistics, spectral energy bands, 3D directional components, autocorrelation periodicity).
        """
        acc_cols = [c for c in df.columns if c.startswith('raw_acc:')]
        gyro_cols = [c for c in df.columns if c.startswith('proc_gyro:') or c.startswith('raw_gyro:')]
        watch_acc_cols = [c for c in df.columns if c.startswith('watch_acceleration:')]
        watch_compass = [c for c in df.columns if c.startswith('watch_compass:')]
        
        selected_cols = acc_cols + gyro_cols + watch_acc_cols + watch_compass
        if not selected_cols:
            selected_cols = [c for c in df.columns if not c.startswith('label:') and c != 'timestamp']
        
        return df[selected_cols].fillna(0.0)

    def load_user_data(self, uuid_file: str) -> Dict[str, Any]:
        """Loads and processes a single user CSV file from ExtraSensory zip."""
        with zipfile.ZipFile(self.zip_path) as z:
            with z.open(uuid_file) as f:
                with gzip.GzipFile(fileobj=f) as gz:
                    df = pd.read_csv(gz)

        timestamps = df['timestamp'].values if 'timestamp' in df.columns else np.arange(len(df)) * 60
        features_df = self._select_sensor_features(df)
        labels, valid_mask = self._extract_7_target_labels(df)
        context_labels = self._extract_optional_labels(df, EXTRASENSORY_CONTEXT_LABELS)
        location_labels = self._extract_optional_labels(df, EXTRASENSORY_LOCATION_LABELS)
        location_features = {
            column: pd.to_numeric(df[column], errors="coerce").to_numpy(dtype=float)
            for column in EXTRASENSORY_LOCATION_FEATURES
            if column in df.columns
        }

        return {
            "uuid": uuid_file.replace(".features_labels.csv.gz", ""),
            "timestamps": timestamps,
            "features_df": features_df,
            "raw_features": features_df.values,
            "labels": labels,
            "valid_mask": valid_mask,
            "context_labels": context_labels,
            "location_labels": location_labels,
            "location_features": location_features,
            "raw_df": df
        }

    @staticmethod
    def _extract_optional_labels(
        df: pd.DataFrame, label_columns: Dict[str, str]
    ) -> Dict[str, np.ndarray]:
        """Retains tri-state context/place labels: positive, negative, or unobserved."""
        return {
            label: (
                pd.to_numeric(df[column], errors="coerce").to_numpy(dtype=float)
                if column in df.columns
                else np.full(len(df), np.nan, dtype=float)
            )
            for label, column in label_columns.items()
        }

    def load_cv_fold_split(
        self,
        fold_idx: int = 0,
        max_sedentary_per_user: int = 3500
    ) -> Dict[str, Any]:
        """
        Loads the official ExtraSensory cross-validation fold (from cv5Folds.zip).
        Fold 0 contains 48 training users and 12 completely unseen test users.
        """
        if not os.path.exists(self.cv_zip_path):
            print(f"[DatasetLoader] Warning: {self.cv_zip_path} not found, falling back to default split.")
            return self.load_dataset_split(num_train_users=40, num_val_users=8, num_test_users=12)

        with zipfile.ZipFile(self.cv_zip_path) as z_cv:
            android_train = z_cv.read(f'cv_5_folds/fold_{fold_idx}_train_android_uuids.txt').decode('utf-8').strip().split('\n')
            iphone_train = z_cv.read(f'cv_5_folds/fold_{fold_idx}_train_iphone_uuids.txt').decode('utf-8').strip().split('\n')
            android_test = z_cv.read(f'cv_5_folds/fold_{fold_idx}_test_android_uuids.txt').decode('utf-8').strip().split('\n')
            iphone_test = z_cv.read(f'cv_5_folds/fold_{fold_idx}_test_iphone_uuids.txt').decode('utf-8').strip().split('\n')

        train_uuids_set = set(u.strip() for u in (android_train + iphone_train) if u.strip())
        test_uuids_set = set(u.strip() for u in (android_test + iphone_test) if u.strip())

        print(f"Loading ExtraSensory Official Fold {fold_idx}: {len(train_uuids_set)} Train Users, {len(test_uuids_set)} Test Users...")

        X_train_list, y_train_list, train_sessions = [], [], []
        X_test_list, y_test_list, test_sessions = [], [], []
        rng = np.random.default_rng(42)
        
        all_files = self.get_available_uuids()

        for fname in all_files:
            uuid = fname.replace('.features_labels.csv.gz', '')
            session_data = self.load_user_data(fname)
            valid = session_data["valid_mask"]
            if np.sum(valid) == 0:
                continue

            X_u = session_data["raw_features"][valid]
            y_u = session_data["labels"][valid]

            if uuid in train_uuids_set:
                train_sessions.append(session_data)
                # Smart balance: keep all rare/dynamic activities, subsample abundant sedentary
                idx_rare = np.where(y_u >= 2)[0]
                idx_sed = np.where(y_u < 2)[0]
                if len(idx_sed) > max_sedentary_per_user:
                    idx_sed = rng.choice(idx_sed, max_sedentary_per_user, replace=False)
                sel = np.concatenate([idx_rare, idx_sed])
                X_train_list.append(X_u[sel])
                y_train_list.append(y_u[sel])

            elif uuid in test_uuids_set:
                test_sessions.append(session_data)
                X_test_list.append(X_u)
                y_test_list.append(y_u)

        # Incorporate archive.zip extra users if available
        if os.path.exists(self.archive_zip_path):
            try:
                with zipfile.ZipFile(self.archive_zip_path) as z_arch:
                    for name in z_arch.namelist():
                        if name.endswith('.csv'):
                            df_extra = pd.read_csv(z_arch.open(name))
                            feat_df = self._select_sensor_features(df_extra)
                            lbls, vld = self._extract_7_target_labels(df_extra)
                            if np.sum(vld) > 0:
                                X_extra = feat_df.values[vld]
                                y_extra = lbls[vld]
                                X_train_list.append(X_extra)
                                y_train_list.append(y_extra)
            except Exception as e:
                print(f"[DatasetLoader] Note on extra archive: {e}")

        X_train = np.vstack(X_train_list)
        y_train = np.concatenate(y_train_list)
        X_test = np.vstack(X_test_list)
        y_test = np.concatenate(y_test_list)

        # Split train into train + validation (e.g. 85% train, 15% val)
        val_split_idx = int(len(X_train) * 0.85)
        perm = rng.permutation(len(X_train))
        X_train, y_train = X_train[perm], y_train[perm]

        X_tr, y_tr = X_train[:val_split_idx], y_train[:val_split_idx]
        X_val, y_val = X_train[val_split_idx:], y_train[val_split_idx:]

        if train_sessions:
            self.feature_columns = list(train_sessions[0]["features_df"].columns)

        X_tr = np.nan_to_num(X_tr, nan=0.0, posinf=0.0, neginf=0.0)
        X_val = np.nan_to_num(X_val, nan=0.0, posinf=0.0, neginf=0.0)
        X_test = np.nan_to_num(X_test, nan=0.0, posinf=0.0, neginf=0.0)

        self.scaler.fit(X_tr)
        self.is_fitted = True

        X_train_scaled = self.scaler.transform(X_tr)
        X_val_scaled = self.scaler.transform(X_val)
        X_test_scaled = self.scaler.transform(X_test)

        print(f"Loaded Dataset -> Train samples: {len(X_train_scaled)}, Val: {len(X_val_scaled)}, Test (Unseen): {len(X_test_scaled)}, Features: {X_train_scaled.shape[1]}")

        return {
            "X_train": X_train_scaled,
            "y_train": y_tr,
            "X_val": X_val_scaled,
            "y_val": y_val,
            "X_test": X_test_scaled,
            "y_test": y_test,
            "train_sessions": train_sessions,
            "val_sessions": train_sessions[:5],
            "test_sessions": test_sessions,
            "feature_columns": self.feature_columns,
            "scaler": self.scaler
        }

    def load_dataset_split(
        self,
        num_train_users: int = 40,
        num_val_users: int = 8,
        num_test_users: int = 12,
        max_samples_per_class: int = 10000
    ) -> Dict[str, Any]:
        """
        Loads multi-user training, validation, and test datasets ensuring
        evaluation occurs on completely unseen users.
        """
        # If cv5Folds.zip exists, use official fold split
        if os.path.exists(self.cv_zip_path):
            return self.load_cv_fold_split(fold_idx=0)

        all_uuids = self.get_available_uuids()
        total_requested = min(num_train_users + num_val_users + num_test_users, len(all_uuids))
        
        train_uuids = all_uuids[:num_train_users]
        val_uuids = all_uuids[num_train_users : num_train_users + num_val_users]
        test_uuids = all_uuids[num_train_users + num_val_users : total_requested]

        print(f"Loading ExtraSensory splits -> Train users: {len(train_uuids)}, Val users: {len(val_uuids)}, Test users: {len(test_uuids)}")

        def aggregate_users(uuid_list: List[str]) -> Tuple[np.ndarray, np.ndarray, List[Dict[str, Any]]]:
            X_list, y_list = [], []
            user_sessions = []
            for u in uuid_list:
                data = self.load_user_data(u)
                user_sessions.append(data)
                valid = data["valid_mask"]
                if np.sum(valid) > 0:
                    X_list.append(data["raw_features"][valid])
                    y_list.append(data["labels"][valid])
            if X_list:
                X = np.vstack(X_list)
                y = np.concatenate(y_list)
            else:
                X = np.empty((0, 0))
                y = np.empty((0,), dtype=int)
            return X, y, user_sessions

        X_train, y_train, train_sessions = aggregate_users(train_uuids)
        X_val, y_val, val_sessions = aggregate_users(val_uuids)
        X_test, y_test, test_sessions = aggregate_users(test_uuids)
        rng = np.random.default_rng(42)

        if train_sessions:
            self.feature_columns = list(train_sessions[0]["features_df"].columns)

        X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
        X_val = np.nan_to_num(X_val, nan=0.0, posinf=0.0, neginf=0.0)
        X_test = np.nan_to_num(X_test, nan=0.0, posinf=0.0, neginf=0.0)

        # Balance classes in training
        if max_samples_per_class > 0 and len(y_train) > 0:
            balanced_idx = []
            for cls_idx in range(len(TARGET_ACTIVITIES)):
                cls_indices = np.where(y_train == cls_idx)[0]
                if len(cls_indices) > max_samples_per_class:
                    selected = rng.choice(cls_indices, size=max_samples_per_class, replace=False)
                    balanced_idx.extend(selected)
                else:
                    balanced_idx.extend(cls_indices)
            balanced_idx = np.array(balanced_idx)
            rng.shuffle(balanced_idx)
            X_train = X_train[balanced_idx]
            y_train = y_train[balanced_idx]

        self.scaler.fit(X_train)
        self.is_fitted = True

        X_train_scaled = self.scaler.transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        X_test_scaled = self.scaler.transform(X_test)

        return {
            "X_train": X_train_scaled,
            "y_train": y_train,
            "X_val": X_val_scaled,
            "y_val": y_val,
            "X_test": X_test_scaled,
            "y_test": y_test,
            "train_sessions": train_sessions,
            "val_sessions": val_sessions,
            "test_sessions": test_sessions,
            "feature_columns": self.feature_columns,
            "scaler": self.scaler
        }

    def synthesize_25hz_raw_signals(
        self,
        duration_sec: float = 6.0,
        activity: str = "walking",
        sampling_rate: int = 25
    ) -> Dict[str, np.ndarray]:
        """
        Synthesizes illustrative, non-measured 25 Hz 6-channel waveforms
        (Acc X/Y/Z in m/s², Gyro X/Y/Z in rad/s) based on biomechanical profiles.
        """
        num_samples = int(duration_sec * sampling_rate)
        t = np.linspace(0, duration_sec, num_samples)

        noise_acc = np.random.normal(0, 0.15, (3, num_samples))
        noise_gyro = np.random.normal(0, 0.08, (3, num_samples))

        if activity == "walking":
            f_step = 1.8  # 1.8 Hz walking cadence
            acc_x = 0.6 * np.sin(2 * np.pi * f_step * t) + noise_acc[0]
            acc_y = 9.81 + 1.8 * np.cos(2 * np.pi * f_step * t) + noise_acc[1]
            acc_z = 0.9 * np.sin(2 * np.pi * f_step * t + np.pi / 4) + noise_acc[2]
            gyro_x = 0.8 * np.sin(2 * np.pi * f_step * t) + noise_gyro[0]
            gyro_y = 0.5 * np.cos(2 * np.pi * f_step * t) + noise_gyro[1]
            gyro_z = 0.4 * np.sin(4 * np.pi * f_step * t) + noise_gyro[2]

        elif activity == "running":
            f_run = 3.0  # 3.0 Hz running cadence
            acc_x = 1.5 * np.sin(2 * np.pi * f_run * t) + noise_acc[0] * 1.5
            acc_y = 9.81 + 4.5 * np.sin(2 * np.pi * f_run * t) + noise_acc[1] * 1.5
            acc_z = 2.2 * np.cos(2 * np.pi * f_run * t) + noise_acc[2] * 1.5
            gyro_x = 2.0 * np.sin(2 * np.pi * f_run * t) + noise_gyro[0]
            gyro_y = 1.2 * np.cos(2 * np.pi * f_run * t) + noise_gyro[1]
            gyro_z = 1.0 * np.sin(2 * np.pi * f_run * t) + noise_gyro[2]

        elif activity == "bicycling":
            f_pedal = 1.3  # 1.3 Hz cadence (78 RPM)
            acc_x = 0.4 * np.sin(2 * np.pi * f_pedal * t) + noise_acc[0]
            acc_y = 9.81 + 0.6 * np.cos(2 * np.pi * f_pedal * t) + noise_acc[1]
            acc_z = 1.2 * np.sin(2 * np.pi * f_pedal * t) + noise_acc[2]
            gyro_x = 0.3 * np.sin(2 * np.pi * f_pedal * t) + noise_gyro[0]
            gyro_y = 1.4 * np.cos(2 * np.pi * f_pedal * t) + noise_gyro[1]
            gyro_z = 0.2 * np.sin(2 * np.pi * f_pedal * t) + noise_gyro[2]

        elif activity == "standing_and_moving":
            acc_x = 0.3 * np.sin(2 * np.pi * 0.8 * t) + noise_acc[0]
            acc_y = 9.81 + 0.4 * np.cos(2 * np.pi * 0.8 * t) + noise_acc[1]
            acc_z = 0.2 * np.sin(2 * np.pi * 0.8 * t) + noise_acc[2]
            gyro_x = 0.2 * np.sin(2 * np.pi * 0.8 * t) + noise_gyro[0]
            gyro_y = 0.2 * np.cos(2 * np.pi * 0.8 * t) + noise_gyro[1]
            gyro_z = 0.1 * np.sin(2 * np.pi * 0.8 * t) + noise_gyro[2]

        elif activity == "standing_in_place":
            acc_x = 0.05 + noise_acc[0] * 0.3
            acc_y = 9.81 + noise_acc[1] * 0.3
            acc_z = 0.02 + noise_acc[2] * 0.3
            gyro_x = noise_gyro[0] * 0.2
            gyro_y = noise_gyro[1] * 0.2
            gyro_z = noise_gyro[2] * 0.2

        elif activity == "sitting":
            acc_x = 0.02 + noise_acc[0] * 0.2
            acc_y = 8.50 + noise_acc[1] * 0.2
            acc_z = 4.80 + noise_acc[2] * 0.2
            gyro_x = noise_gyro[0] * 0.15
            gyro_y = noise_gyro[1] * 0.15
            gyro_z = noise_gyro[2] * 0.15

        elif activity == "lying_down":
            acc_x = 0.01 + noise_acc[0] * 0.1
            acc_y = 0.10 + noise_acc[1] * 0.1
            acc_z = 9.81 + noise_acc[2] * 0.1
            gyro_x = noise_gyro[0] * 0.08
            gyro_y = noise_gyro[1] * 0.08
            gyro_z = noise_gyro[2] * 0.08

        else:
            acc_x = noise_acc[0]
            acc_y = 9.81 + noise_acc[1]
            acc_z = noise_acc[2]
            gyro_x = noise_gyro[0]
            gyro_y = noise_gyro[1]
            gyro_z = noise_gyro[2]

        return {
            "time": t,
            "Acc X": acc_x,
            "Acc Y": acc_y,
            "Acc Z": acc_z,
            "Gyro X": gyro_x,
            "Gyro Y": gyro_y,
            "Gyro Z": gyro_z
        }
