"""Data processing utilities for sensor data analysis."""


import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler


class SensorDataProcessor:
    """Handles preprocessing of sensor data for behavior detection."""

    def __init__(self) -> None:
        self.scalers: dict[str, StandardScaler] = {}
        self.label_encoders: dict[str, LabelEncoder] = {}
        self.feature_columns: list[str] | None = None

    def preprocess_sensor_data(
        self, df: pd.DataFrame, fit: bool = True
    ) -> pd.DataFrame:
        """Preprocess sensor data with scaling and feature extraction."""
        df_processed = df.copy()

        # Handle missing values
        df_processed = self._handle_missing_values(df_processed)

        # Extract time-based features if timestamp exists
        if "timestamp" in df_processed.columns:
            df_processed = self._extract_time_features(df_processed)

        # Scale numerical features
        numeric_columns = df_processed.select_dtypes(include=[np.number]).columns
        if fit:
            self.feature_columns = list(numeric_columns)

        for col in numeric_columns:
            if col not in ["id", "target", "behavior"]:  # Skip ID and target columns
                if fit:
                    if col not in self.scalers:
                        self.scalers[col] = StandardScaler()
                    df_processed[col] = self.scalers[col].fit_transform(
                        df_processed[[col]]
                    )
                else:
                    if col in self.scalers:
                        df_processed[col] = self.scalers[col].transform(
                            df_processed[[col]]
                        )

        return df_processed

    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in sensor data."""
        # Forward fill for sensor readings (common in time series)
        sensor_columns = [
            col
            for col in df.columns
            if any(x in col.lower() for x in ["accel", "gyro", "mag", "sensor"])
        ]

        for col in sensor_columns:
            if col in df.columns:
                df[col] = df[col].ffill().bfill()

        # Fill remaining numerical columns with median
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            if df[col].isnull().sum() > 0:
                df[col] = df[col].fillna(df[col].median())

        return df

    def _extract_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract time-based features from timestamp."""
        if "timestamp" not in df.columns:
            return df

        df["timestamp"] = pd.to_datetime(df["timestamp"])

        # Extract time components
        df["hour"] = df["timestamp"].dt.hour
        df["minute"] = df["timestamp"].dt.minute
        df["second"] = df["timestamp"].dt.second
        df["day_of_week"] = df["timestamp"].dt.dayofweek
        df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)

        # Extract cyclical features
        df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
        df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
        df["minute_sin"] = np.sin(2 * np.pi * df["minute"] / 60)
        df["minute_cos"] = np.cos(2 * np.pi * df["minute"] / 60)

        return df

    def extract_sensor_features(
        self, df: pd.DataFrame, window_size: int = 10
    ) -> pd.DataFrame:
        """Extract statistical features from sensor data using rolling windows."""
        feature_df = df.copy()

        # Find sensor columns (accelerometer, gyroscope, etc.)
        sensor_columns = [
            col
            for col in df.columns
            if any(x in col.lower() for x in ["accel", "gyro", "mag", "x", "y", "z"])
        ]

        for col in sensor_columns:
            if col in df.columns:
                # Rolling statistics
                feature_df[f"{col}_mean"] = df[col].rolling(window_size).mean()
                feature_df[f"{col}_std"] = df[col].rolling(window_size).std()
                feature_df[f"{col}_min"] = df[col].rolling(window_size).min()
                feature_df[f"{col}_max"] = df[col].rolling(window_size).max()
                feature_df[f"{col}_range"] = (
                    feature_df[f"{col}_max"] - feature_df[f"{col}_min"]
                )

                # Magnitude features for 3D sensors
                if any(axis in col.lower() for axis in ["_x", "_y", "_z"]):
                    base_name = col.rsplit("_", 1)[0]  # Remove _x, _y, _z suffix
                    if (
                        f"{base_name}_x" in df.columns
                        and f"{base_name}_y" in df.columns
                        and f"{base_name}_z" in df.columns
                    ):
                        magnitude_col = f"{base_name}_magnitude"
                        if magnitude_col not in feature_df.columns:
                            feature_df[magnitude_col] = np.sqrt(
                                df[f"{base_name}_x"] ** 2
                                + df[f"{base_name}_y"] ** 2
                                + df[f"{base_name}_z"] ** 2
                            )

        return feature_df

    def create_sequences(
        self, df: pd.DataFrame, sequence_length: int = 50, step_size: int = 10
    ) -> tuple[np.ndarray, np.ndarray | None]:
        """Create sequences for time series modeling."""
        if self.feature_columns is None:
            raise ValueError(
                "Feature columns not defined. Run preprocess_sensor_data first."
            )

        features = df[self.feature_columns].values
        targets = df["target"].values if "target" in df.columns else None

        X_sequences = []
        y_sequences = []

        for i in range(0, len(features) - sequence_length + 1, step_size):
            X_sequences.append(features[i : i + sequence_length])
            if targets is not None:
                y_sequences.append(
                    targets[i + sequence_length - 1]
                )  # Use last target in sequence

        X_array = np.array(X_sequences)
        y_array = np.array(y_sequences) if targets is not None else None

        return X_array, y_array


def load_and_preprocess_data(
    data_path: str, processor: SensorDataProcessor | None = None
) -> tuple[pd.DataFrame, SensorDataProcessor]:
    """Load and preprocess sensor data from file."""
    if processor is None:
        processor = SensorDataProcessor()

    df = pd.read_csv(data_path)
    print(f"Loaded data with shape: {df.shape}")

    # Preprocess the data
    df_processed = processor.preprocess_sensor_data(df, fit=True)

    # Extract additional sensor features
    df_processed = processor.extract_sensor_features(df_processed)

    print(f"Processed data with shape: {df_processed.shape}")
    feature_count = len(processor.feature_columns) if processor.feature_columns else 0
    print(f"Feature columns: {feature_count}")

    return df_processed, processor
