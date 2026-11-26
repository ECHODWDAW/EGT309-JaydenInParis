
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.data_cleaning import DataCleaner


class FeatureEngineer:
    """Generate and select features for predictive maintenance."""

    def __init__(self, config):
        self.config = config

    def transform(self, df):
        """Generate features and remove unwanted columns."""

        df = df.copy()

        timestamp_col = self.config["data"]["timestamp"]
        machine_col = self.config["data"]["machine_id"]
        target_col = self.config["data"]["target"]

        # Generate cyclical time features.
        hour = (
            df[timestamp_col].dt.hour
            + df[timestamp_col].dt.minute / 60
        )

        df["hour_sin"] = np.sin(2 * np.pi * hour / 24)
        df["hour_cos"] = np.cos(2 * np.pi * hour / 24)

        day = df[timestamp_col].dt.day_of_week

        df["day_sin"] = np.sin(2 * np.pi * day / 7)
        df["day_cos"] = np.cos(2 * np.pi * day / 7)

        # Identify missing sensor readings.
        df["temperature_missing"] = (
            df["temperature"].isna().astype(int)
        )

        df["vibration_missing"] = (
            df["vibration"].isna().astype(int)
        )

        # Select approved input features.
        selected_columns = (
            self.config["features"]["sensor_columns"]
            + [
                machine_col,
                "hour_sin",
                "hour_cos",
                "day_sin",
                "day_cos",
                "temperature_missing",
                "vibration_missing",
                timestamp_col,
                target_col
            ]
        )

        df = df[selected_columns].copy()

        print("\n--- Feature Engineering ---")
        print("Cyclical time features generated.")
        print("Missing-value indicators generated.")
        print("Potential data leakage columns excluded.")

        return df


class DataPreprocessor:
    """Prepare shared training and testing datasets."""

    def __init__(self, config_path="config/config.yaml"):

        self.project_root = Path(__file__).resolve().parents[1]

        with open(
            self.project_root / config_path,
            "r",
            encoding="utf-8"
        ) as file:
            self.config = yaml.safe_load(file)

        self.config_path = config_path

        self.feature_engineer = FeatureEngineer(self.config)

        self.preprocessor = None
        self.feature_names = None

    def split_data(self, df):
        """Split data chronologically without timestamp overlap."""

        timestamp_col = self.config["data"]["timestamp"]
        target_col = self.config["data"]["target"]

        test_size = self.config["split"]["test_size"]

        if not 0 < test_size < 1:
            raise ValueError("test_size must be between 0 and 1.")

        if self.config["split"]["method"] != "chronological":
            raise ValueError("Unsupported splitting method.")

        df = df.sort_values(timestamp_col).reset_index(drop=True)

        split_index = int(len(df) * (1 - test_size))
        split_timestamp = df.iloc[split_index][timestamp_col]

        train_df = df[
            df[timestamp_col] < split_timestamp
        ].copy()

        test_df = df[
            df[timestamp_col] >= split_timestamp
        ].copy()

        if train_df.empty or test_df.empty:
            raise ValueError("Training or testing dataset is empty.")

        X_train = train_df.drop(
            columns=[target_col, timestamp_col]
        )

        X_test = test_df.drop(
            columns=[target_col, timestamp_col]
        )

        y_train = train_df[target_col]
        y_test = test_df[target_col]

        print("\n--- Dataset Splitting ---")
        print(f"Training records: {len(X_train)}")
        print(f"Testing records: {len(X_test)}")
        print(f"Split timestamp: {split_timestamp}")

        print("\nTraining target distribution:")
        print(y_train.value_counts())

        print("\nTesting target distribution:")
        print(y_test.value_counts())

        return X_train, X_test, y_train, y_test

    def build_preprocessor(self):
        """Build the shared preprocessing transformer."""

        numerical_features = (
            self.config["features"]["sensor_columns"]
            + [
                "hour_sin",
                "hour_cos",
                "day_sin",
                "day_cos"
            ]
        )

        binary_features = [
            "temperature_missing",
            "vibration_missing"
        ]

        categorical_features = [
            self.config["data"]["machine_id"]
        ]

        numerical_pipeline = Pipeline(
            steps=[
                (
                    "imputer",
                    SimpleImputer(
                        strategy=self.config["cleaning"][
                            "temperature_imputation"
                        ]
                    )
                ),
                ("scaler", StandardScaler())
            ]
        )

        categorical_pipeline = Pipeline(
            steps=[
                (
                    "encoder",
                    OneHotEncoder(
                        handle_unknown="ignore",
                        sparse_output=False
                    )
                )
            ]
        )

        self.preprocessor = ColumnTransformer(
            transformers=[
                (
                    "numerical",
                    numerical_pipeline,
                    numerical_features
                ),
                (
                    "categorical",
                    categorical_pipeline,
                    categorical_features
                ),
                (
                    "binary",
                    "passthrough",
                    binary_features
                )
            ],
            remainder="drop",
            verbose_feature_names_out=False
        )

        return self.preprocessor

    def transform_data(self, X_train, X_test):
        """Fit on training data and transform both datasets."""

        if self.preprocessor is None:
            self.build_preprocessor()

        # Learn preprocessing parameters from training data only.
        X_train_processed = self.preprocessor.fit_transform(
            X_train
        )

        # Apply the same parameters to unseen testing data.
        X_test_processed = self.preprocessor.transform(
            X_test
        )

        self.feature_names = (
            self.preprocessor.get_feature_names_out()
        )

        X_train_processed = pd.DataFrame(
            X_train_processed,
            columns=self.feature_names,
            index=X_train.index
        )

        X_test_processed = pd.DataFrame(
            X_test_processed,
            columns=self.feature_names,
            index=X_test.index
        )

        print("\n--- Preprocessing Results ---")

        print(f"Training shape: {X_train_processed.shape}")
        print(f"Testing shape: {X_test_processed.shape}")

        print(
            "Training missing values:",
            X_train_processed.isnull().sum().sum()
        )

        print(
            "Testing missing values:",
            X_test_processed.isnull().sum().sum()
        )

        print(f"Total features: {len(self.feature_names)}")

        return X_train_processed, X_test_processed

    def prepare_raw_data(self):
        """Clean, engineer and split data before fitting."""

        cleaner = DataCleaner(self.config_path)
        cleaned_df = cleaner.run()

        engineered_df = self.feature_engineer.transform(
            cleaned_df
        )

        return self.split_data(engineered_df)

    def run(self):
        """Execute the complete preprocessing pipeline."""

        X_train, X_test, y_train, y_test = (
            self.prepare_raw_data()
        )

        self.build_preprocessor()

        X_train, X_test = self.transform_data(
            X_train,
            X_test
        )

        print("\nPreprocessing completed successfully.")

        return X_train, X_test, y_train, y_test


if __name__ == "__main__":

    processor = DataPreprocessor()

    X_train, X_test, y_train, y_test = processor.run()

    print("\n--- Final Dataset ---")

    print(f"X_train: {X_train.shape}")
    print(f"X_test: {X_test.shape}")
    print(f"y_train: {y_train.shape}")
    print(f"y_test: {y_test.shape}")

    print("\nFeature names:")
    print(processor.feature_names)

