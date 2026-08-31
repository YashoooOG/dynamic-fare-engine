"""
Cancellation & Price Elasticity Risk Classifier Module.
Estimates the probability of a customer cancelling or rejecting a trip quote
as a function of surge multiplier, trip distance, wait times, and operational conditions.
"""

import os
import logging
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier

from src.models.evaluate import evaluate_classification

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_CANCELLATION_CONFIG = {
    "model_type": "xgboost",
    "n_estimators": 80,
    "learning_rate": 0.08,
    "max_depth": 4,
    "random_state": 42
}


def generate_cancellation_training_data(n_samples: int = 50000, seed: int = 42) -> tuple:
    """
    Synthesize empirical rider conversion / cancellation dataset based on
    microeconomic price elasticity curves, wait times, and weather friction.
    """
    np.random.seed(seed)
    
    surge = np.random.uniform(1.0, 3.5, size=n_samples)
    distance = np.random.exponential(scale=3.5, size=n_samples).clip(0.5, 20.0)
    wait_time = np.random.gamma(shape=3.0, scale=2.0, size=n_samples).clip(1.0, 25.0)
    traffic_sev = np.random.choice([1, 2, 3, 4], size=n_samples, p=[0.3, 0.4, 0.2, 0.1])
    weather_sev = np.random.choice([1, 2, 3, 4], size=n_samples, p=[0.5, 0.25, 0.15, 0.1])
    is_new_user = np.random.choice([0, 1], size=n_samples, p=[0.75, 0.25])
    
    # Base estimated fare
    fare = (3.5 + 1.85 * distance + 0.35 * (distance * 3.0)) * surge
    
    # Latent cancellation probability via logistic link function
    # Higher surge, higher wait time, bad weather, and new users increase cancellation probability
    z = (
        -2.50
        + 1.35 * (surge - 1.0)
        + 0.09 * wait_time
        + 0.12 * (traffic_sev - 1)
        + 0.15 * (weather_sev - 1)
        + 0.30 * is_new_user
    )
    prob_cancel = 1.0 / (1.0 + np.exp(-z))
    cancelled = (np.random.rand(n_samples) < prob_cancel).astype(int)
    
    df = pd.DataFrame({
        "surge_multiplier": surge,
        "distance": distance,
        "fare": fare,
        "wait_time_min": wait_time,
        "traffic_severity": traffic_sev,
        "weather_severity": weather_sev,
        "is_new_user": is_new_user,
        "cancelled": cancelled
    })
    
    feature_cols = [
        "surge_multiplier", "distance", "fare", "wait_time_min",
        "traffic_severity", "weather_severity", "is_new_user"
    ]
    X = df[feature_cols]
    y = df["cancelled"]
    
    return X, y


class CancellationModel:
    """Classifier to predict customer drop-off / cancellation probability."""
    
    def __init__(self, config: dict = None):
        self.config = {**DEFAULT_CANCELLATION_CONFIG, **(config or {})}
        self.model = self._init_model()
        self.feature_names = None
        self.is_fitted = False
        
    def _init_model(self):
        mtype = self.config.get("model_type", "xgboost").lower()
        if mtype == "xgboost":
            return XGBClassifier(
                n_estimators=self.config.get("n_estimators", 80),
                learning_rate=self.config.get("learning_rate", 0.08),
                max_depth=self.config.get("max_depth", 4),
                random_state=self.config.get("random_state", 42),
                eval_metric="logloss"
            )
        elif mtype == "logistic":
            return LogisticRegression(C=1.0, max_iter=500, random_state=42)
        else:
            return GradientBoostingClassifier(
                n_estimators=self.config.get("n_estimators", 80),
                learning_rate=self.config.get("learning_rate", 0.08),
                max_depth=self.config.get("max_depth", 4),
                random_state=self.config.get("random_state", 42)
            )
            
    def fit(self, X: pd.DataFrame, y: pd.Series):
        """Fit cancellation probability classifier."""
        self.feature_names = list(X.columns) if hasattr(X, "columns") else None
        logger.info(f"Fitting cancellation model ({self.config.get('model_type')}) on {len(X):,} samples...")
        self.model.fit(X, y)
        self.is_fitted = True
        logger.info("Cancellation risk model training completed.")
        return self
        
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict cancellation probability (class 1)."""
        if not self.is_fitted:
            raise ValueError("Model is not fitted yet.")
        probs = self.model.predict_proba(X)
        return probs[:, 1]
        
    def predict(self, X: pd.DataFrame, threshold: float = 0.5) -> np.ndarray:
        """Predict binary cancellation class."""
        probs = self.predict_proba(X)
        return (probs >= threshold).astype(int)
        
    def evaluate(self, X: pd.DataFrame, y: pd.Series, split_name: str = "Test") -> dict:
        """Evaluate classification metrics."""
        probs = self.predict_proba(X)
        metrics = evaluate_classification(y, probs, model_name=f"Cancel_{self.config.get('model_type')}_{split_name}")
        logger.info(f"[{split_name} Cancellation Eval] ROC-AUC: {metrics['roc_auc']:.4f} | Log-Loss: {metrics['log_loss']:.4f} | Acc: {metrics['accuracy']:.4f} | F1: {metrics['f1']:.4f}")
        return metrics
        
    def save(self, filepath: str) -> str:
        """Persist model artifact."""
        out_path = Path(filepath)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, filepath)
        logger.info(f"Saved cancellation model to {filepath}")
        return filepath
        
    @classmethod
    def load(cls, filepath: str) -> "CancellationModel":
        """Load model from disk."""
        if not Path(filepath).exists():
            raise FileNotFoundError(f"Model file not found at: {filepath}")
        obj = joblib.load(filepath)
        logger.info(f"Loaded cancellation model from {filepath}")
        return obj


def train_cancellation_model(X_train: pd.DataFrame, y_train: pd.Series, config: dict = None) -> CancellationModel:
    """Convenience function to fit CancellationModel."""
    model = CancellationModel(config)
    model.fit(X_train, y_train)
    return model
