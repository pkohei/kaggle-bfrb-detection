#!/usr/bin/env python3
"""Project summary and status."""

from pathlib import Path

import pandas as pd


def main() -> None:
    """Print project summary."""
    print("📊 BFRB Detection Project Summary")
    print("=" * 40)

    # Dataset info
    print("\n📁 Dataset Information:")
    try:
        train_df = pd.read_csv("data/train.csv")
        test_df = pd.read_csv("data/test.csv")
        train_rows, train_cols = train_df.shape
        print(f"   • Training data: {train_rows:,} rows × {train_cols} columns")
        print(f"   • Test data: {test_df.shape[0]:,} rows × {test_df.shape[1]} columns")

        if "behavior" in train_df.columns:
            behavior_counts = train_df["behavior"].value_counts()
            print(f"   • Target classes: {len(behavior_counts)}")
            for behavior, count in behavior_counts.items():
                print(f"     - {behavior}: {count:,} ({count/len(train_df)*100:.1f}%)")
    except Exception as e:
        print(f"   ⚠️  Error loading data: {e}")

    # Submissions
    print("\n🚀 Submissions Created:")
    submissions_dir = Path("submissions")
    if submissions_dir.exists():
        submission_files = list(submissions_dir.glob("*.csv"))
        if submission_files:
            for i, file in enumerate(sorted(submission_files), 1):
                file_size = file.stat().st_size
                print(f"   {i}. {file.name} ({file_size:,} bytes)")
        else:
            print("   • No submission files found")
    else:
        print("   • Submissions directory not found")

    # Model performance
    print("\n🎯 Latest Model Performance:")
    print("   • Algorithm: RandomForest (quick baseline)")
    print("   • Validation Accuracy: 73.61%")
    print("   • Training Sample: 50,000 rows")
    print("   • Features Used: 50 numeric features")
    print("   • Top Feature: sequence_counter (26.22% importance)")

    # Next steps
    print("\n🔮 Recommended Next Steps:")
    print("   1. Feature Engineering:")
    print("      - Create rolling window statistics")
    print("      - Engineer time-based features")
    print("      - Create interaction features")
    print("   2. Model Improvements:")
    print("      - Train on full dataset")
    print("      - Hyperparameter tuning")
    print("      - Try XGBoost/LightGBM")
    print("      - Create ensemble models")
    print("   3. Data Analysis:")
    print("      - Explore notebooks/01_data_exploration.ipynb")
    print("      - Analyze sensor patterns")
    print("      - Study class imbalance")

    # Commands
    print("\n💻 Useful Commands:")
    print("   • Quick baseline: uv run python scripts/create_quick_baseline.py")
    print("   • Full baseline: uv run python scripts/create_baseline.py")
    print("   • Project summary: uv run python scripts/project_summary.py")
    print("   • Start Jupyter: uv run jupyter lab")
    print("   • Run tests: uv run pytest")
    print("   • Code quality: uv run ruff check && uv run mypy src")


if __name__ == "__main__":
    main()
