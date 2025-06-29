"""Kaggle API utilities for CMI competition."""

import os
import zipfile
from pathlib import Path

import kaggle
import pandas as pd


class KaggleCompetition:
    """Handles Kaggle competition data download and submission."""

    def __init__(self, competition_name: str = "cmi-detect-behavior-with-sensor-data"):
        self.competition_name = competition_name
        self.data_dir = Path("data")
        self.data_dir.mkdir(exist_ok=True)

    def download_data(self, force: bool = False) -> None:
        """Download competition data if not already present."""
        if not force and self._data_exists():
            print("Data already exists. Use force=True to re-download.")
            return

        print(f"Downloading data for {self.competition_name}...")
        kaggle.api.competition_download_files(
            self.competition_name, path=str(self.data_dir), quiet=False
        )

        # Extract zip files
        for zip_file in self.data_dir.glob("*.zip"):
            print(f"Extracting {zip_file.name}...")
            with zipfile.ZipFile(zip_file, "r") as zip_ref:
                zip_ref.extractall(self.data_dir)
            zip_file.unlink()  # Remove zip file after extraction

    def _data_exists(self) -> bool:
        """Check if competition data already exists."""
        return len(list(self.data_dir.glob("*.csv"))) > 0

    def load_train_data(self) -> pd.DataFrame:
        """Load training data."""
        train_path = self.data_dir / "train.csv"
        if not train_path.exists():
            raise FileNotFoundError(
                "Training data not found. Run download_data() first."
            )
        return pd.read_csv(train_path)

    def load_test_data(self) -> pd.DataFrame:
        """Load test data."""
        test_path = self.data_dir / "test.csv"
        if not test_path.exists():
            raise FileNotFoundError("Test data not found. Run download_data() first.")
        return pd.read_csv(test_path)

    def load_sample_submission(self) -> pd.DataFrame:
        """Load sample submission format."""
        sample_path = self.data_dir / "sample_submission.csv"
        if not sample_path.exists():
            raise FileNotFoundError(
                "Sample submission not found. Run download_data() first."
            )
        return pd.read_csv(sample_path)

    def submit_predictions(
        self, submission_file: str, message: str = "Submission"
    ) -> None:
        """Submit predictions to Kaggle."""
        submission_path = Path(submission_file)
        if not submission_path.exists():
            raise FileNotFoundError(f"Submission file not found: {submission_file}")

        print(f"Submitting {submission_file} to {self.competition_name}...")
        kaggle.api.competition_submit(
            file_name=str(submission_path),
            message=message,
            competition=self.competition_name,
        )
        print("Submission completed!")


def setup_kaggle_credentials() -> None:
    """Setup Kaggle API credentials."""
    kaggle_dir = Path.home() / ".kaggle"
    kaggle_dir.mkdir(exist_ok=True)

    credentials_file = kaggle_dir / "kaggle.json"

    if not credentials_file.exists():
        print("Kaggle credentials not found.")
        print("Please download kaggle.json from https://www.kaggle.com/account")
        print(f"and place it in {kaggle_dir}")
        return

    # Set proper permissions
    os.chmod(credentials_file, 0o600)
    print("Kaggle credentials configured successfully!")


if __name__ == "__main__":
    # Setup and download data
    setup_kaggle_credentials()

    competition = KaggleCompetition()
    competition.download_data()

    # Load and preview data
    try:
        train_df = competition.load_train_data()
        print(f"Training data shape: {train_df.shape}")
        print(train_df.head())
    except FileNotFoundError as e:
        print(f"Error loading data: {e}")
