#!/usr/bin/env python3
"""Create quick baseline model for BFRB detection competition."""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


def main() -> None:
    """Create quick baseline model."""
    print("🚀 Creating Quick Baseline Model for BFRB Detection")
    print("=" * 55)

    # Load data
    print("1. Loading data...")
    train_df = pd.read_csv("data/train.csv")
    test_df = pd.read_csv("data/test.csv")

    print(f"   ✓ Train data shape: {train_df.shape}")
    print(f"   ✓ Test data shape: {test_df.shape}")

    # Use sample for quick training
    sample_size = min(50000, len(train_df))
    train_sample = train_df.sample(n=sample_size, random_state=42)
    print(f"   ✓ Using sample of {sample_size} rows for quick training")

    # Find target and features
    target_col = "behavior"
    numeric_cols = train_sample.select_dtypes(include=[np.number]).columns.tolist()
    feature_cols = [col for col in numeric_cols if col not in [target_col, "id"]][
        :50
    ]  # Use first 50 features

    print(f"   ✓ Target: {target_col}")
    print(f"   ✓ Using {len(feature_cols)} features")

    # Prepare data
    print("\n2. Preparing data...")
    X = train_sample[feature_cols].fillna(0).values

    # Encode target
    le = LabelEncoder()
    y = le.fit_transform(train_sample[target_col].values)

    print(f"   ✓ Classes: {le.classes_}")
    print(f"   ✓ Class distribution: {pd.Series(y).value_counts().to_dict()}")

    # Split data
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"   ✓ Train: {X_train.shape}, Val: {X_val.shape}")

    # Train simple model
    print("\n3. Training RandomForest model...")
    model = RandomForestClassifier(
        n_estimators=50,  # Reduced for speed
        max_depth=10,
        random_state=42,
        n_jobs=-1,
    )

    model.fit(X_train, y_train)

    # Evaluate
    print("\n4. Evaluating model...")
    y_pred = model.predict(X_val)
    accuracy = accuracy_score(y_val, y_pred)

    print(f"   ✓ Validation Accuracy: {accuracy:.4f}")
    print("\n   Classification Report:")
    print(classification_report(y_val, y_pred, target_names=le.classes_))

    # Create submission
    print("\n5. Creating submission...")

    # Prepare test data
    X_test = test_df[feature_cols].fillna(0).values
    test_ids = (
        test_df["id"].values if "id" in test_df.columns else np.arange(len(test_df))
    )

    # Get predictions
    y_test_pred = model.predict(X_test)

    # Convert back to original labels
    predicted_labels = le.inverse_transform(y_test_pred)

    # Create submission DataFrame
    submission_df = pd.DataFrame({"id": test_ids, "behavior": predicted_labels})

    # Save submission
    from pathlib import Path

    Path("submissions").mkdir(exist_ok=True)

    import datetime

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    submission_file = f"submissions/quick_baseline_{timestamp}.csv"

    submission_df.to_csv(submission_file, index=False)

    print(f"   ✓ Submission saved: {submission_file}")
    print(f"   ✓ Submission shape: {submission_df.shape}")
    print("\n   Sample predictions:")
    print(submission_df.head(10))

    # Feature importance
    print("\n6. Feature importance (top 10):")
    feature_importance = (
        pd.DataFrame(
            {"feature": feature_cols, "importance": model.feature_importances_}
        )
        .sort_values("importance", ascending=False)
        .head(10)
    )

    for _, row in feature_importance.iterrows():
        print(f"   {row['feature']}: {row['importance']:.4f}")

    print("\n🎉 Quick baseline completed!")
    print(f"   • Model accuracy: {accuracy:.4f}")
    print(f"   • Submission file: {submission_file}")
    print("   • Ready to submit to Kaggle!")


if __name__ == "__main__":
    main()
