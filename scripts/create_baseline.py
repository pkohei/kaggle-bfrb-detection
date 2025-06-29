#!/usr/bin/env python3
"""Create baseline model for BFRB detection competition."""

import numpy as np
import pandas as pd
from bfrb.evaluation import ModelEvaluator, SubmissionGenerator
from bfrb.models import LightGBMModel, ModelEnsemble, RandomForestModel, XGBoostModel


def main() -> None:
    """Create baseline model."""
    print("🎯 Creating Baseline Model for BFRB Detection")
    print("=" * 50)

    # Load and explore data
    print("1. Loading and exploring data...")
    train_df = pd.read_csv("data/train.csv")
    test_df = pd.read_csv("data/test.csv")
    train_demo = pd.read_csv("data/train_demographics.csv")
    test_demo = pd.read_csv("data/test_demographics.csv")

    print(f"   ✓ Train data shape: {train_df.shape}")
    print(f"   ✓ Test data shape: {test_df.shape}")
    print(f"   ✓ Train demographics shape: {train_demo.shape}")
    print(f"   ✓ Test demographics shape: {test_demo.shape}")

    # Check for target column
    target_col = None
    for col in ["behavior", "target", "label"]:
        if col in train_df.columns:
            target_col = col
            break

    if target_col is None:
        print("   ⚠️  No target column found. Checking available columns...")
        print(f"   Available columns: {list(train_df.columns)}")
        # Use the last column as target if no obvious target found
        target_col = train_df.columns[-1]
        print(f"   Using '{target_col}' as target column")
    else:
        print(f"   ✓ Target column found: {target_col}")

    # Basic data preprocessing
    print("\n2. Preprocessing data...")

    # Simple preprocessing for baseline
    # Remove non-numeric columns except target
    numeric_cols = train_df.select_dtypes(include=[np.number]).columns.tolist()
    if target_col in numeric_cols and target_col != "id":
        feature_cols = [
            col for col in numeric_cols if col != target_col and col != "id"
        ]
    else:
        feature_cols = [col for col in numeric_cols if col != "id"]

    print(f"   ✓ Using {len(feature_cols)} numeric features")

    # Prepare training data
    X = train_df[feature_cols].fillna(0).values
    if target_col in train_df.columns:
        # Encode target if it's categorical
        from sklearn.preprocessing import LabelEncoder

        le = LabelEncoder()
        y = le.fit_transform(train_df[target_col].values)
        print(f"   ✓ Target encoded. Classes: {le.classes_}")
    else:
        y = np.zeros(len(train_df))

    # Prepare test data
    X_test = test_df[feature_cols].fillna(0).values
    test_ids = (
        test_df["id"].values if "id" in test_df.columns else np.arange(len(test_df))
    )

    print(f"   ✓ Training data prepared: {X.shape}")
    print(f"   ✓ Test data prepared: {X_test.shape}")
    print(f"   ✓ Target distribution: {pd.Series(y).value_counts().to_dict()}")

    # Split data for validation
    print("\n3. Creating train/validation split...")
    from sklearn.model_selection import train_test_split

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"   ✓ Train set: {X_train.shape}")
    print(f"   ✓ Validation set: {X_val.shape}")

    # Create models
    print("\n4. Training baseline models...")
    n_classes = len(np.unique(y))
    print(f"   ✓ Number of classes: {n_classes}")

    models = {
        "LightGBM": LightGBMModel(
            objective="multiclass",
            num_class=n_classes,
            metric="multi_logloss",
            num_leaves=31,
            learning_rate=0.1,
            feature_fraction=0.9,
            bagging_fraction=0.8,
            verbose=-1,
        ),
        "XGBoost": XGBoostModel(
            objective="multi:softprob",
            num_class=n_classes,
            eval_metric="mlogloss",
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
        ),
        "RandomForest": RandomForestModel(
            n_estimators=100, max_depth=10, random_state=42
        ),
    }

    # Train individual models
    trained_models = {}
    evaluator = ModelEvaluator()

    for name, model in models.items():
        print(f"   Training {name}...")
        try:
            if name == "RandomForest":
                model.train(X_train, y_train)
            else:
                model.train(X_train, y_train, X_val, y_val)
            trained_models[name] = model

            # Quick evaluation
            val_metrics = evaluator.evaluate_model(model, X_val, y_val, name)
            print(f"   ✓ {name} - Accuracy: {val_metrics['accuracy']:.4f}")
        except Exception as e:
            print(f"   ⚠️  {name} training failed: {e}")

    if not trained_models:
        print("   ❌ No models trained successfully!")
        return

    # Create ensemble
    print("\n5. Creating ensemble model...")
    ensemble = ModelEnsemble(trained_models)
    ensemble.is_trained = True  # Models are already trained

    # Evaluate ensemble
    ensemble_pred = ensemble.predict(X_val)
    ensemble_accuracy = np.mean(ensemble_pred == y_val)
    print(f"   ✓ Ensemble accuracy: {ensemble_accuracy:.4f}")

    # Generate submissions
    print("\n6. Generating submissions...")
    submission_gen = SubmissionGenerator()

    submission_files = {}

    # Individual model submissions
    for name, model in trained_models.items():
        try:
            submission_file = submission_gen.create_submission(
                model,
                test_df[feature_cols].fillna(0),
                test_ids,
                f"baseline_{name.lower()}",
            )
            submission_files[name] = submission_file
            print(f"   ✓ {name} submission created")
        except Exception as e:
            print(f"   ⚠️  {name} submission failed: {e}")

    # Ensemble submission
    try:
        ensemble_file = submission_gen.create_ensemble_submission(
            trained_models,
            test_df[feature_cols].fillna(0),
            test_ids,
            submission_name="baseline_ensemble",
        )
        submission_files["Ensemble"] = ensemble_file
        print("   ✓ Ensemble submission created")
    except Exception as e:
        print(f"   ⚠️  Ensemble submission failed: {e}")

    # Model comparison
    print("\n7. Model comparison...")
    if len(trained_models) > 1:
        comparison_df = evaluator.compare_models(trained_models, X_val, y_val)
        print("\n📊 Model Performance Summary:")
        print(comparison_df.to_string(index=False))

    # Summary
    print("\n🎉 Baseline model creation completed!")
    print(f"\n📁 Created {len(submission_files)} submission files:")
    for name, file_path in submission_files.items():
        print(f"   • {name}: {file_path}")

    print("\n🚀 Next steps:")
    print("1. Review model performance in results/model_comparison.csv")
    print("2. Submit your best model to Kaggle")
    print("3. Iterate on feature engineering and model tuning")
    print("4. Use notebooks for detailed analysis")


if __name__ == "__main__":
    main()
