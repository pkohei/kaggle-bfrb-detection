"""Machine learning models for behavior detection."""


from typing import Any, Protocol

import lightgbm as lgb
import numpy as np
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold, cross_val_score


class MLModel(Protocol):
    """Protocol for machine learning models."""

    def predict(self, X: np.ndarray) -> np.ndarray:
        ...

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        ...


class BehaviorDetectionModel:
    """Base class for behavior detection models."""

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self.model: Any = None
        self.is_trained = False
        self.feature_importance_: Any = None

    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        """Train the model."""
        raise NotImplementedError("Subclasses must implement train method")

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        if not self.is_trained or self.model is None:
            raise ValueError("Model must be trained before making predictions")
        return self.model.predict(X)  # type: ignore

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict probabilities."""
        if not self.is_trained or self.model is None:
            raise ValueError("Model must be trained before making predictions")
        if hasattr(self.model, "predict_proba"):
            return self.model.predict_proba(X)  # type: ignore
        else:
            raise ValueError("Model does not support probability prediction")

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> dict[str, object]:
        """Evaluate model performance."""
        predictions = self.predict(X)
        accuracy = accuracy_score(y, predictions)

        return {"accuracy": float(accuracy), "model_name": self.model_name}

    def cross_validate(
        self, X: np.ndarray, y: np.ndarray, cv: int = 5
    ) -> dict[str, float]:
        """Perform cross-validation."""
        if not self.is_trained:
            raise ValueError("Model must be trained before cross-validation")

        cv_scores = cross_val_score(
            self.model,
            X,
            y,
            cv=StratifiedKFold(n_splits=cv, shuffle=True, random_state=42),
        )

        return {
            "cv_mean": cv_scores.mean(),
            "cv_std": cv_scores.std(),
            "cv_scores": cv_scores.tolist(),
        }


class LightGBMModel(BehaviorDetectionModel):
    """LightGBM model for behavior detection."""

    def __init__(self, **params: Any) -> None:
        super().__init__("LightGBM")

        default_params = {
            "objective": "binary",
            "metric": "binary_logloss",
            "boosting_type": "gbdt",
            "num_leaves": 31,
            "learning_rate": 0.05,
            "feature_fraction": 0.9,
            "bagging_fraction": 0.8,
            "bagging_freq": 5,
            "verbose": -1,
            "random_state": 42,
        }
        default_params.update(params)
        self.params = default_params

    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
    ) -> None:
        """Train LightGBM model."""
        train_data = lgb.Dataset(X, label=y)
        valid_sets = [train_data]

        if X_val is not None and y_val is not None:
            valid_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
            valid_sets.append(valid_data)

        trained_model = lgb.train(
            self.params,
            train_data,
            valid_sets=valid_sets,
            num_boost_round=1000,
            callbacks=[lgb.early_stopping(100), lgb.log_evaluation(100)],
        )
        self.model = trained_model
        self.is_trained = True
        self.feature_importance_: Any = trained_model.feature_importance()

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        if not self.is_trained or self.model is None:
            raise ValueError("Model must be trained before making predictions")
        probas = self.model.predict(X)  # type: ignore
        return (probas > 0.5).astype(int)  # type: ignore

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict probabilities."""
        if not self.is_trained or self.model is None:
            raise ValueError("Model must be trained before making predictions")
        probas = self.model.predict(X)  # type: ignore
        return np.vstack([1 - probas, probas]).T  # type: ignore


class XGBoostModel(BehaviorDetectionModel):
    """XGBoost model for behavior detection."""

    def __init__(self, **params: Any) -> None:
        super().__init__("XGBoost")

        default_params = {
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "max_depth": 6,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "random_state": 42,
        }
        default_params.update(params)
        self.params = default_params

    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
    ) -> None:
        """Train XGBoost model."""
        dtrain = xgb.DMatrix(X, label=y)
        evals = [(dtrain, "train")]

        if X_val is not None and y_val is not None:
            dval = xgb.DMatrix(X_val, label=y_val)
            evals.append((dval, "val"))

        trained_model = xgb.train(
            self.params,
            dtrain,
            num_boost_round=1000,
            evals=evals,
            early_stopping_rounds=100,
            verbose_eval=100,
        )
        self.model = trained_model
        self.is_trained = True
        self.feature_importance_: Any = trained_model.get_score(
            importance_type="weight"
        )

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        if not self.is_trained or self.model is None:
            raise ValueError("Model must be trained before making predictions")
        dtest = xgb.DMatrix(X)
        probas = self.model.predict(dtest)  # type: ignore
        return (probas > 0.5).astype(int)  # type: ignore

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict probabilities."""
        if not self.is_trained or self.model is None:
            raise ValueError("Model must be trained before making predictions")
        dtest = xgb.DMatrix(X)
        probas = self.model.predict(dtest)  # type: ignore
        return np.vstack([1 - probas, probas]).T  # type: ignore


class RandomForestModel(BehaviorDetectionModel):
    """Random Forest model for behavior detection."""

    def __init__(self, **params: Any) -> None:
        super().__init__("RandomForest")

        default_params = {
            "n_estimators": 100,
            "max_depth": 10,
            "min_samples_split": 5,
            "min_samples_leaf": 2,
            "random_state": 42,
            "n_jobs": -1,
        }
        default_params.update(params)
        self.model: RandomForestClassifier = RandomForestClassifier(**default_params)

    def train(self, X: np.ndarray, y: np.ndarray, **kwargs: Any) -> None:
        """Train Random Forest model."""
        if isinstance(self.model, RandomForestClassifier):
            self.model.fit(X, y)
            self.is_trained = True
            self.feature_importance_: Any = self.model.feature_importances_


class ModelEnsemble:
    """Ensemble of multiple models for improved performance."""

    def __init__(self, models: dict[str, BehaviorDetectionModel]) -> None:
        self.models = models
        self.weights: list[float] | None = None
        self.is_trained = False

    def train(self, X: np.ndarray, y: np.ndarray, **kwargs: Any) -> None:
        """Train all models in the ensemble."""
        for name, model in self.models.items():
            print(f"Training {name}...")
            model.train(X, y, **kwargs)

        self.is_trained = True
        print("Ensemble training completed!")

    def predict(self, X: np.ndarray, use_weights: bool = True) -> np.ndarray:
        """Make ensemble predictions."""
        if not self.is_trained:
            raise ValueError("Ensemble must be trained before making predictions")

        prediction_list = []
        for model in self.models.values():
            pred = model.predict_proba(X)[:, 1]  # Get positive class probabilities
            prediction_list.append(pred)

        predictions = np.array(prediction_list)

        if use_weights and self.weights is not None:
            ensemble_pred = np.average(predictions, axis=0, weights=self.weights)
        else:
            ensemble_pred = np.mean(predictions, axis=0)

        return (ensemble_pred > 0.5).astype(int)  # type: ignore

    def predict_proba(self, X: np.ndarray, use_weights: bool = True) -> np.ndarray:
        """Predict ensemble probabilities."""
        if not self.is_trained:
            raise ValueError("Ensemble must be trained before making predictions")

        prediction_list = []
        for model in self.models.values():
            pred = model.predict_proba(X)[:, 1]  # Get positive class probabilities
            prediction_list.append(pred)

        predictions = np.array(prediction_list)

        if use_weights and self.weights is not None:
            ensemble_pred = np.average(predictions, axis=0, weights=self.weights)
        else:
            ensemble_pred = np.mean(predictions, axis=0)

        return np.vstack([1 - ensemble_pred, ensemble_pred]).T

    def set_weights(self, weights: dict[str, float]) -> None:
        """Set weights for ensemble models."""
        model_names = list(self.models.keys())
        weight_list = [weights.get(name, 1.0) for name in model_names]
        total_weight = sum(weight_list)
        self.weights = [w / total_weight for w in weight_list]  # Normalize weights
