"""
Demand & Ride Fare Forecasting Model Module.
Wraps regression estimators (XGBoost, Random Forest, Ridge) for predicting
trip fares and operational demand metrics.
"""

import os
import logging
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

from src.models.evaluate import evaluate_regression

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_CONFIG = {
    "model_type": "xgboost",
    "n_estimators": 120,
    "learning_rate": 0.08,
    "max_depth": 6,
    "random_state": 42,
    "n_jobs": -1
}


class DemandForecastModel:
    """Wrapper class for demand and fare price forecasting regression models."""
    
    def __init__(self, config: dict = None):
        self.config = {**DEFAULT_CONFIG, **(config or {})}
        self.model = self._init_model()
        self.feature_names = None
        self.is_fitted = False
        
    def _init_model(self):
        mtype = self.config.get("model_type", "xgboost").lower()
        if mtype == "xgboost":
            return XGBRegressor(
                n_estimators=self.config.get("n_estimators", 120),
                learning_rate=self.config.get("learning_rate", 0.08),
                max_depth=self.config.get("max_depth", 6),
                random_state=self.config.get("random_state", 42),
                n_jobs=self.config.get("n_jobs", -1)
            )
        elif mtype == "random_forest":
            return RandomForestRegressor(
                n_estimators=self.config.get("n_estimators", 80),
                max_depth=self.config.get("max_depth", 10),
                random_state=self.config.get("random_state", 42),
                n_jobs=self.config.get("n_jobs", -1)
            )
        elif mtype == "ridge":
            return Ridge(alpha=self.config.get("alpha", 1.0))
        else:
            raise ValueError(f"Unsupported model type: {mtype}")
            
    def fit(self, X: pd.DataFrame, y: pd.Series):
        """Fit estimator on training features and target labels."""
        self.feature_names = list(X.columns) if hasattr(X, "columns") else None
        logger.info(f"Fitting {self.config.get('model_type')} demand model on {len(X):,} samples...")
        self.model.fit(X, y)
        self.is_fitted = True
        logger.info("Demand model training completed.")
        return self
        
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Generate predictions for feature matrix X."""
        if not self.is_fitted:
            raise ValueError("Model is not fitted yet. Call fit() or load() first.")
        return self.model.predict(X)
        
    def evaluate(self, X: pd.DataFrame, y: pd.Series, split_name: str = "Test") -> dict:
        """Predict and evaluate regression metrics against true targets."""
        preds = self.predict(X)
        metrics = evaluate_regression(y, preds, model_name=f"Demand_{self.config.get('model_type')}_{split_name}")
        logger.info(f"[{split_name} Evaluation] MAE: ${metrics['mae']:.3f} | RMSE: ${metrics['rmse']:.3f} | R²: {metrics['r2']:.4f}")
        return metrics
        
    def get_feature_importances(self) -> pd.Series:
        """Return relative feature importance scores."""
        if hasattr(self.model, "feature_importances_") and self.feature_names:
            return pd.Series(self.model.feature_importances_, index=self.feature_names).sort_values(ascending=False)
        return pd.Series(dtype=float)
        
    def save(self, filepath: str) -> str:
        """Persist trained model artifact using joblib."""
        out_path = Path(filepath)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, filepath)
        logger.info(f"Saved demand forecast model to {filepath}")
        return filepath
        
    @classmethod
    def load(cls, filepath: str) -> "DemandForecastModel":
        """Load persisted model artifact from disk."""
        if not Path(filepath).exists():
            raise FileNotFoundError(f"Model file not found at: {filepath}")
        obj = joblib.load(filepath)
        logger.info(f"Loaded demand forecast model from {filepath}")
        return obj


def train_demand_model(X_train: pd.DataFrame, y_train: pd.Series, config: dict = None) -> DemandForecastModel:
    """Convenience function to initialize and fit DemandForecastModel."""
    model = DemandForecastModel(config)
    model.fit(X_train, y_train)
    return model
