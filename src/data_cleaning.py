
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


class DataCleaner:
    """Load, validate and clean machine sensor data."""

    def __init__(self, config_path="config/config.yaml"):
        self.project_root = Path(__file__).resolve().parents[1]

        with open(
            self.project_root / config_path,
            "r",
            encoding="utf-8"
        ) as file:
            self.config = yaml.safe_load(file)

        self.data_path = (
            self.project_root / self.config["data"]["path"]
        )

        self.df = None
        self.original_rows = 0
        self.cleaning_report = {}

    def load_data(self):
        """Load the original CSV dataset."""

        if not self.data_path.exists():
            raise FileNotFoundError(
                f"Dataset not found: {self.data_path}"
            )

        self.df = pd.read_csv(self.data_path)
        self.original_rows = len(self.df)

        print("\n--- Dataset Loading ---")
        print("Dataset loaded successfully.")
        print(f"Number of rows: {self.df.shape[0]}")
        print(f"Number of columns: {self.df.shape[1]}")

        return self.df

    def validate_data(self):
        """Validate required columns and inspect data quality."""

        if self.df is None:
            raise ValueError("Dataset has not been loaded.")

        required_columns = (
            self.config["features"]["sensor_columns"]
            + self.config["features"]["exclude_columns"]
            + [
                self.config["data"]["target"],
                self.config["data"]["timestamp"],
                self.config["data"]["machine_id"]
            ]
        )

        missing_columns = [
            col for col in required_columns
            if col not in self.df.columns
        ]

        if missing_columns:
            raise ValueError(
                f"Missing required columns: {missing_columns}"
            )

        print("\n--- Data Validation ---")
        print("All required columns are present.")

        print("\nMissing values:")
        print(self.df.isnull().sum())

        print(f"\nDuplicate rows: {self.df.duplicated().sum()}")

        print("\nData types:")
        print(self.df.dtypes)

    def convert_data_types(self):
        """Convert timestamps, identifiers and sensor data."""

        timestamp_col = self.config["data"]["timestamp"]
        machine_col = self.config["data"]["machine_id"]
        target_col = self.config["data"]["target"]

        self.df[timestamp_col] = pd.to_datetime(
            self.df[timestamp_col],
            errors="coerce"
        )

        self.df[machine_col] = self.df[machine_col].astype(
            "string"
        )

        for col in self.config["features"]["sensor_columns"]:
            self.df[col] = pd.to_numeric(
                self.df[col],
                errors="coerce"
            )

            self.df[col] = self.df[col].replace(
                [np.inf, -np.inf],
                np.nan
            )

        self.df[target_col] = pd.to_numeric(
            self.df[target_col],
            errors="coerce"
        )

        print("\nData types converted successfully.")

    def remove_invalid_records(self):
        """Remove records with invalid identifiers or targets."""

        timestamp_col = self.config["data"]["timestamp"]
        machine_col = self.config["data"]["machine_id"]
        target_col = self.config["data"]["target"]

        before = len(self.df)

        self.df = self.df.dropna(
            subset=[timestamp_col, machine_col, target_col]
        )

        self.df = self.df[
            self.df[target_col].isin([0, 1])
        ].copy()

        self.df[target_col] = self.df[target_col].astype(int)

        removed = before - len(self.df)

        self.cleaning_report["invalid_records_removed"] = removed

        print(f"Invalid records removed: {removed}")

    def remove_duplicates(self):
        """Remove identical duplicate records."""

        before = len(self.df)

        if self.config["cleaning"]["remove_duplicates"]:
            self.df = self.df.drop_duplicates().copy()

        removed = before - len(self.df)

        self.cleaning_report["duplicates_removed"] = removed

        print(f"Duplicate rows removed: {removed}")

        key_columns = [
            self.config["data"]["machine_id"],
            self.config["data"]["timestamp"]
        ]

        duplicate_keys = self.df.duplicated(
            subset=key_columns,
            keep=False
        ).sum()

        self.cleaning_report["duplicate_key_rows"] = int(
            duplicate_keys
        )

        if duplicate_keys > 0:
            raise ValueError(
                "Conflicting machine-timestamp records detected. "
                "Review these records before continuing."
            )

    def clean_vibration(self):
        """Identify and handle invalid vibration readings."""

        timestamp_col = self.config["data"]["timestamp"]

        negative_mask = self.df["vibration"] < 0

        error_minutes = self.config["cleaning"][
            "vibration_error_minutes"
        ]

        suspected_mask = (
            negative_mask
            & self.df[timestamp_col].dt.minute.isin(
                error_minutes
            )
        )

        suspected_count = int(suspected_mask.sum())

        # Do not assume a sign-inversion error without
        # confirming the pattern through EDA.
        corrected_count = 0

        correction_enabled = self.config["cleaning"].get(
            "correct_vibration",
            False
        )

        correction_verified = self.config["cleaning"].get(
            "vibration_sign_verified",
            False
        )

        if correction_enabled and correction_verified:
            self.df.loc[suspected_mask, "vibration"] = (
                self.df.loc[suspected_mask, "vibration"].abs()
            )

            corrected_count = suspected_count

        # Treat remaining negative vibration values as missing.
        remaining_negative = self.df["vibration"] < 0

        invalid_count = int(remaining_negative.sum())

        self.df.loc[remaining_negative, "vibration"] = np.nan

        self.cleaning_report[
            "suspected_vibration_errors"
        ] = suspected_count

        self.cleaning_report[
            "vibration_corrected"
        ] = corrected_count

        self.cleaning_report[
            "invalid_vibration_to_missing"
        ] = invalid_count

        print(f"Suspected vibration errors: {suspected_count}")
        print(f"Vibration readings corrected: {corrected_count}")
        print(f"Invalid vibration set to missing: {invalid_count}")

    def clean_sensor_values(self):
        """Handle physically invalid sensor readings."""

        invalid_masks = {
            "humidity": (
                (self.df["humidity"] < 0)
                | (self.df["humidity"] > 100)
            ),
            "pressure": self.df["pressure"] < 0,
            "energy_consumption": (
                self.df["energy_consumption"] < 0
            )
        }

        for column, mask in invalid_masks.items():
            count = int(mask.sum())

            self.df.loc[mask, column] = np.nan

            self.cleaning_report[
                f"invalid_{column}_to_missing"
            ] = count

            print(f"Invalid {column} readings: {count}")

    def sort_data(self):
        """Sort records chronologically."""

        timestamp_col = self.config["data"]["timestamp"]
        machine_col = self.config["data"]["machine_id"]

        self.df = self.df.sort_values(
            by=[timestamp_col, machine_col]
        ).reset_index(drop=True)

        print("Dataset sorted chronologically.")

    def generate_summary(self):
        """Display the final data cleaning summary."""

        print("\n--- Data Cleaning Summary ---")

        print(f"Original rows: {self.original_rows}")
        print(f"Remaining rows: {len(self.df)}")
        print(f"Rows removed: {self.original_rows - len(self.df)}")

        print("\nCleaning operations:")

        for operation, count in self.cleaning_report.items():
            print(f"{operation}: {count}")

        print("\nRemaining missing values:")
        print(self.df.isnull().sum())

        target_col = self.config["data"]["target"]

        print("\nTarget distribution:")
        print(self.df[target_col].value_counts())

        print("\nData cleaning completed successfully.")

    def run(self):
        """Execute the complete data cleaning pipeline."""

        self.cleaning_report = {}

        self.load_data()
        self.validate_data()

        print("\n--- Data Cleaning ---")

        self.convert_data_types()
        self.remove_invalid_records()
        self.remove_duplicates()
        self.clean_vibration()
        self.clean_sensor_values()
        self.sort_data()
        self.generate_summary()

        return self.df


if __name__ == "__main__":
    cleaner = DataCleaner()
    cleaned_df = cleaner.run()
