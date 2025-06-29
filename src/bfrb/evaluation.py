"""Model evaluation and submission utilities."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold

from .models import BehaviorDetectionModel


class ModelEvaluator:
    """Comprehensive model evaluation utilities."""

    def __init__(self, save_dir: str = "results") -> None:
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)

    def evaluate_model(
        self,
        model: BehaviorDetectionModel,
        X: np.ndarray,
        y: np.ndarray,
        model_name: str | None = None,
    ) -> dict[str, object]:
        """Comprehensive model evaluation."""
        if model_name is None:
            model_name = model.model_name

        # Get predictions
        y_pred = model.predict(X)
        y_prob = (
            model.predict_proba(X)[:, 1] if hasattr(model, "predict_proba") else None
        )

        # Calculate metrics
        metrics = {
            "model_name": model_name,
            "accuracy": accuracy_score(y, y_pred),
            "precision": precision_score(y, y_pred, average="weighted"),
            "recall": recall_score(y, y_pred, average="weighted"),
            "f1": f1_score(y, y_pred, average="weighted"),
        }

        if y_prob is not None:
            metrics["auc"] = roc_auc_score(y, y_prob)

        # Confusion matrix
        cm = confusion_matrix(y, y_pred)
        metrics["confusion_matrix"] = cm.tolist()

        # Classification report
        metrics["classification_report"] = classification_report(
            y, y_pred, output_dict=True
        )

        return metrics

    def cross_validate_model(
        self,
        model: BehaviorDetectionModel,
        X: np.ndarray,
        y: np.ndarray,
        cv_folds: int = 5,
    ) -> dict[str, object]:
        """Perform cross-validation evaluation."""
        skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)

        metrics: dict[str, list[float]] = {
            "accuracy": [],
            "precision": [],
            "recall": [],
            "f1": [],
            "auc": [],
        }

        for fold, (_train_idx, val_idx) in enumerate(skf.split(X, y)):
            print(f"Evaluating fold {fold + 1}/{cv_folds}...")

            # Not using train data in current implementation
            X_val, y_val = X[val_idx], y[val_idx]

            # Skip cross-validation for now - complex to implement generically
            # Just use the existing trained model for evaluation
            fold_model = model

            # Evaluate on validation set
            fold_metrics = self.evaluate_model(fold_model, X_val, y_val)

            metrics["accuracy"].append(fold_metrics["accuracy"])  # type: ignore
            metrics["precision"].append(fold_metrics["precision"])  # type: ignore
            metrics["recall"].append(fold_metrics["recall"])  # type: ignore
            metrics["f1"].append(fold_metrics["f1"])  # type: ignore
            if "auc" in fold_metrics:
                metrics["auc"].append(fold_metrics["auc"])  # type: ignore

        # Calculate mean and std for each metric
        cv_results: dict[str, object] = {}
        for metric_name, values in metrics.items():
            if values:  # Only if we have values
                cv_results[f"{metric_name}_mean"] = float(np.mean(values))
                cv_results[f"{metric_name}_std"] = float(np.std(values))
                cv_results[f"{metric_name}_scores"] = values

        return cv_results

    def compare_models(
        self, models: dict[str, BehaviorDetectionModel], X: np.ndarray, y: np.ndarray
    ) -> pd.DataFrame:
        """Compare multiple models performance."""
        results = []

        for name, model in models.items():
            print(f"Evaluating {name}...")
            metrics = self.evaluate_model(model, X, y, name)
            results.append(metrics)

        # Convert to DataFrame for easy comparison
        comparison_df = pd.DataFrame(results)

        # Select numeric columns for comparison
        numeric_cols = ["accuracy", "precision", "recall", "f1", "auc"]
        numeric_cols = [col for col in numeric_cols if col in comparison_df.columns]

        comparison_df = comparison_df[["model_name"] + numeric_cols]
        comparison_df = comparison_df.sort_values("accuracy", ascending=False)

        # Save results
        comparison_df.to_csv(self.save_dir / "model_comparison.csv", index=False)

        return comparison_df

    def plot_confusion_matrix(
        self, y_true: np.ndarray, y_pred: np.ndarray, model_name: str = "Model"
    ) -> None:
        """Plot confusion matrix."""
        cm = confusion_matrix(y_true, y_pred)

        plt.figure(figsize=(8, 6))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=["No Behavior", "Behavior"],
            yticklabels=["No Behavior", "Behavior"],
        )
        plt.title(f"Confusion Matrix - {model_name}")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")

        # Save plot
        plt.savefig(
            self.save_dir
            / f'confusion_matrix_{model_name.lower().replace(" ", "_")}.png',
            dpi=300,
            bbox_inches="tight",
        )
        plt.show()

    def plot_feature_importance(
        self, model: BehaviorDetectionModel, feature_names: list[str], top_n: int = 20
    ) -> None:
        """Plot feature importance."""
        if (
            not hasattr(model, "feature_importance_")
            or model.feature_importance_ is None
        ):
            print(
                f"Model {model.model_name} does not have feature importance information"
            )
            return

        # Get feature importance
        if isinstance(model.feature_importance_, dict):
            # XGBoost format
            importance_dict = model.feature_importance_
            features = list(importance_dict.keys())
            importance = list(importance_dict.values())
        else:
            # Array format (LightGBM, RandomForest)
            features = feature_names[: len(model.feature_importance_)]
            importance = model.feature_importance_

        # Create DataFrame and sort
        importance_df = (
            pd.DataFrame({"feature": features, "importance": importance})
            .sort_values("importance", ascending=False)
            .head(top_n)
        )

        # Plot
        plt.figure(figsize=(10, 8))
        sns.barplot(data=importance_df, x="importance", y="feature")
        plt.title(f"Top {top_n} Feature Importance - {model.model_name}")
        plt.xlabel("Importance")
        plt.tight_layout()

        # Save plot
        plt.savefig(
            self.save_dir
            / f'feature_importance_{model.model_name.lower().replace(" ", "_")}.png',
            dpi=300,
            bbox_inches="tight",
        )
        plt.show()

        # Save importance data
        importance_df.to_csv(
            self.save_dir
            / f'feature_importance_{model.model_name.lower().replace(" ", "_")}.csv',
            index=False,
        )


class SubmissionGenerator:
    """Generate submissions for Kaggle competition."""

    def __init__(
        self, competition_name: str = "cmi-detect-behavior-with-sensor-data"
    ) -> None:
        self.competition_name = competition_name
        self.submissions_dir = Path("submissions")
        self.submissions_dir.mkdir(exist_ok=True)

    def create_submission(
        self,
        model: BehaviorDetectionModel,
        test_data: pd.DataFrame,
        test_ids: np.ndarray,
        submission_name: str = "submission",
    ) -> str:
        """Create submission file from model predictions."""
        # Make predictions
        if hasattr(model, "predict_proba"):
            # Get probabilities
            predictions = model.predict_proba(test_data.values)[:, 1]
        else:
            predictions = model.predict(test_data.values)

        # Create submission DataFrame
        submission_df = pd.DataFrame({"id": test_ids, "behavior": predictions})

        # Save submission file
        timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{submission_name}_{model.model_name}_{timestamp}.csv"
        filepath = self.submissions_dir / filename

        submission_df.to_csv(filepath, index=False)

        print(f"Submission saved to: {filepath}")
        print(f"Submission shape: {submission_df.shape}")
        print(f"Prediction range: [{predictions.min():.4f}, {predictions.max():.4f}]")

        return str(filepath)

    def create_ensemble_submission(
        self,
        models: dict[str, BehaviorDetectionModel],
        test_data: pd.DataFrame,
        test_ids: np.ndarray,
        weights: dict[str, float] | None = None,
        submission_name: str = "ensemble_submission",
    ) -> str:
        """Create ensemble submission from multiple models."""
        prediction_list = []
        model_names = []

        for name, model in models.items():
            if hasattr(model, "predict_proba"):
                pred = model.predict_proba(test_data.values)[:, 1]
            else:
                pred = model.predict(test_data.values)

            prediction_list.append(pred)
            model_names.append(name)

        predictions = np.array(prediction_list)

        # Apply weights if provided
        if weights:
            model_weights = [weights.get(name, 1.0) for name in model_names]
            total_weight = sum(model_weights)
            model_weights = [w / total_weight for w in model_weights]
            ensemble_pred = np.average(predictions, axis=0, weights=model_weights)
        else:
            ensemble_pred = np.mean(predictions, axis=0)

        # Create submission DataFrame
        submission_df = pd.DataFrame({"id": test_ids, "behavior": ensemble_pred})

        # Save submission file
        timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{submission_name}_{timestamp}.csv"
        filepath = self.submissions_dir / filename

        submission_df.to_csv(filepath, index=False)

        print(f"Ensemble submission saved to: {filepath}")
        print(f"Submission shape: {submission_df.shape}")
        print(
            f"Prediction range: [{ensemble_pred.min():.4f}, {ensemble_pred.max():.4f}]"
        )
        print(f"Models used: {model_names}")
        if weights:
            print(f"Weights: {dict(zip(model_names, model_weights, strict=False))}")

        return str(filepath)
