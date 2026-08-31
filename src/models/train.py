"""
Main Model Training Pipeline Orchestrator.
Loads preprocessed training splits from data/processed/, trains demand forecasting,
delivery friction, and cancellation risk models, evaluates performance,
and saves the final .joblib artifacts to models/.
"""

import sys
import logging
from pathlib import Path
import pandas as pd
import yaml

from src.features.build_features import build_all
from src.models.demand_forecast import DemandForecastModel
from src.models.cancellation_model import CancellationModel, generate_cancellation_training_data
from src.models.evaluate import evaluate_regression, evaluate_classification
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def load_config(config_path: str = "config.yaml") -> dict:
    """Load configuration YAML if present."""
    path = Path(config_path)
    if path.exists():
        with open(path, "r") as f:
            return yaml.safe_load(f) or {}
    return {}


def train_all_models(config_path: str = "config.yaml") -> dict:
    """
    Execute full end-to-end model training workflow:
    1. Check/load processed datasets
    2. Train and evaluate ride demand/fare model
    3. Train and evaluate delivery delay model
    4. Train and evaluate customer cancellation model
    5. Save all models to models/*.joblib
    """
    config = load_config(config_path)
    models_dir = Path(config.get("paths", {}).get("models_dir", "models"))
    models_dir.mkdir(parents=True, exist_ok=True)
    processed_dir = Path(config.get("paths", {}).get("processed_dir", "data/processed"))
    
    # 1. Check if processed splits exist, else build
    ride_train_file = processed_dir / "ride_train.parquet"
    if not ride_train_file.exists():
        logger.info("Processed data not found. Executing build_all()...")
        build_all()
        
    logger.info("Loading processed training and test splits...")
    
    # =========================================================================
    # A. Ride Demand / Fare Forecasting Model
    # =========================================================================
    logger.info("--- [1/3] Training Ride Fare / Demand Model ---")
    ride_train = pd.read_parquet(processed_dir / "ride_train.parquet")
    ride_val = pd.read_parquet(processed_dir / "ride_val.parquet")
    ride_test = pd.read_parquet(processed_dir / "ride_test.parquet")
    
    target_col = "price"
    X_ride_train = ride_train.drop(columns=[target_col])
    y_ride_train = ride_train[target_col]
    X_ride_val = ride_val.drop(columns=[target_col])
    y_ride_val = ride_val[target_col]
    X_ride_test = ride_test.drop(columns=[target_col])
    y_ride_test = ride_test[target_col]
    
    ride_model_cfg = config.get("models", {}).get("ride_demand", {
        "model_type": "xgboost",
        "n_estimators": 120,
        "learning_rate": 0.08,
        "max_depth": 6
    })
    
    ride_demand_model = DemandForecastModel(ride_model_cfg)
    ride_demand_model.fit(X_ride_train, y_ride_train)
    
    # Evaluate
    val_ride_metrics = ride_demand_model.evaluate(X_ride_val, y_ride_val, split_name="Validation")
    test_ride_metrics = ride_demand_model.evaluate(X_ride_test, y_ride_test, split_name="Test")
    
    # Save artifacts
    ride_model_path = models_dir / "demand_model_ride.joblib"
    ride_demand_model.save(str(ride_model_path))
    # Save alias for notebook compatibility
    ride_demand_model.save(str(models_dir / "ride_fare_xgb.joblib"))
    
    # =========================================================================
    # B. Delivery Operational Delay / Friction Model
    # =========================================================================
    logger.info("--- [2/3] Training Delivery Operational Friction Model ---")
    deliv_train = pd.read_parquet(processed_dir / "delivery_train.parquet")
    deliv_val = pd.read_parquet(processed_dir / "delivery_val.parquet")
    deliv_test = pd.read_parquet(processed_dir / "delivery_test.parquet")
    
    deliv_target = "delivery_delay"
    X_deliv_train = deliv_train.drop(columns=[deliv_target])
    y_deliv_train = deliv_train[deliv_target]
    X_deliv_val = deliv_val.drop(columns=[deliv_target])
    y_deliv_val = deliv_val[deliv_target]
    X_deliv_test = deliv_test.drop(columns=[deliv_target])
    y_deliv_test = deliv_test[deliv_target]
    
    deliv_model_cfg = config.get("models", {}).get("delivery_friction", {
        "model_type": "xgboost",
        "n_estimators": 100,
        "learning_rate": 0.08,
        "max_depth": 5
    })
    
    deliv_model = DemandForecastModel(deliv_model_cfg)
    deliv_model.fit(X_deliv_train, y_deliv_train)
    
    val_deliv_metrics = deliv_model.evaluate(X_deliv_val, y_deliv_val, split_name="Validation")
    test_deliv_metrics = deliv_model.evaluate(X_deliv_test, y_deliv_test, split_name="Test")
    
    deliv_model_path = models_dir / "demand_model_delivery.joblib"
    deliv_model.save(str(deliv_model_path))
    deliv_model.save(str(models_dir / "delivery_friction_xgb.joblib"))
    
    # =========================================================================
    # C. Cancellation / Customer Acceptance Risk Classifier
    # =========================================================================
    logger.info("--- [3/3] Training Customer Cancellation Risk Model ---")
    X_cancel, y_cancel = generate_cancellation_training_data(n_samples=60000, seed=42)
    
    X_c_train_val, X_c_test, y_c_train_val, y_c_test = train_test_split(
        X_cancel, y_cancel, test_size=0.15, random_state=42
    )
    X_c_train, X_c_val, y_c_train, y_c_val = train_test_split(
        X_c_train_val, y_c_train_val, test_size=0.15 / 0.85, random_state=42
    )
    
    cancel_model_cfg = config.get("models", {}).get("cancellation", {
        "model_type": "xgboost",
        "n_estimators": 80,
        "learning_rate": 0.08,
        "max_depth": 4
    })
    
    cancellation_model = CancellationModel(cancel_model_cfg)
    cancellation_model.fit(X_c_train, y_c_train)
    
    val_cancel_metrics = cancellation_model.evaluate(X_c_val, y_c_val, split_name="Validation")
    test_cancel_metrics = cancellation_model.evaluate(X_c_test, y_c_test, split_name="Test")
    
    cancel_model_path = models_dir / "cancellation_model.joblib"
    cancellation_model.save(str(cancel_model_path))
    
    logger.info("============================================================")
    logger.info("MODEL TRAINING PIPELINE COMPLETED SUCCESSFULLY")
    logger.info(f"Ride Model Test R²: {test_ride_metrics['r2']:.4f} (MAE: ${test_ride_metrics['mae']:.2f})")
    logger.info(f"Delivery Model Test R²: {test_deliv_metrics['r2']:.4f} (MAE: {test_deliv_metrics['mae']:.2f} mins)")
    logger.info(f"Cancellation Model Test ROC-AUC: {test_cancel_metrics['roc_auc']:.4f}")
    logger.info(f"Saved artifacts to {models_dir}/")
    logger.info("============================================================")
    
    return {
        "ride_model": ride_demand_model,
        "delivery_model": deliv_model,
        "cancellation_model": cancellation_model,
        "metrics": {
            "ride": {"val": val_ride_metrics, "test": test_ride_metrics},
            "delivery": {"val": val_deliv_metrics, "test": test_deliv_metrics},
            "cancellation": {"val": val_cancel_metrics, "test": test_cancel_metrics}
        }
    }


if __name__ == "__main__":
    cfg_file = sys.argv[1] if len(sys.argv) > 1 else "config.yaml"
    train_all_models(cfg_file)
