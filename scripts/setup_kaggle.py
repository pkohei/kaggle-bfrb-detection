#!/usr/bin/env python3
"""Setup script for Kaggle CMI competition environment."""

from pathlib import Path

from bfrb.kaggle_utils import KaggleCompetition, setup_kaggle_credentials


def main() -> None:
    """Main setup function."""
    print("🏁 Setting up Kaggle CMI Competition Environment")
    print("=" * 50)

    # Setup Kaggle credentials
    print("1. Setting up Kaggle credentials...")
    setup_kaggle_credentials()

    # Create necessary directories
    print("2. Creating project directories...")
    directories = ["data", "notebooks", "results", "submissions", "models"]

    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"   ✓ Created {directory}/")

    # Download competition data
    print("3. Downloading competition data...")
    try:
        competition = KaggleCompetition()
        competition.download_data()
        print("   ✓ Data downloaded successfully!")
    except Exception as e:
        print(f"   ⚠️  Data download failed: {e}")
        print("   Please ensure your Kaggle credentials are set up correctly.")

    # Verify setup
    print("\n4. Verifying setup...")

    # Check if data files exist
    data_files = ["train.csv", "test.csv", "sample_submission.csv"]
    for file in data_files:
        if (Path("data") / file).exists():
            print(f"   ✓ {file} found")
        else:
            print(f"   ⚠️  {file} not found")

    print("\n🎉 Setup completed!")
    print("\nNext steps:")
    print("1. Open notebooks/01_data_exploration.ipynb to explore the data")
    print("2. Run notebooks/02_model_training.ipynb to train models")
    print("3. Submit your best model to Kaggle!")

    print("\nUseful commands:")
    print("- uv run jupyter lab  # Start Jupyter Lab")
    print("- uv run python -m bfrb.kaggle_utils  # Test Kaggle connection")
    print("- uv run pytest  # Run tests")


if __name__ == "__main__":
    main()
