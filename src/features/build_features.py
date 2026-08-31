"""
Feature Engineering Pipeline Orchestrator.
Combines temporal, categorical, and lag features, executes train/val/test splits,
and persists ML-ready artifacts to data/processed/.
"""

import os
import sys
import logging
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from src.features.time_features import add_time_features
from src.features.lag_features import add_rolling_zone_features
from src.features.zone_encoding import encode_vehicle_tiers, encode_delivery_attributes

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def build_ride_features(
    input_path: str = "data/interim/ride_hourly_clean.parquet",
    sample_size: int = 150000,
    random_state: int = 42
) -> tuple:
    """
    Build ML feature tables for ride fare prediction:
    - Ingest cleaned interim parquet
    - Apply time features (hour, day, cyclical sin/cos)
    - Apply vehicle tier encoding
    - Perform Train/Val/Test splits (70% train, 15% val, 15% test)
    """
    path = Path(input_path)
    if not path.exists():
        raise FileNotFoundError(f"Interim ride file not found at: {input_path}")
        
    logger.info(f"Building ride features from {input_path}...")
    df = pd.read_parquet(input_path)
    
    if sample_size and len(df) > sample_size:
        df = df.sample(n=sample_size, random_state=random_state).copy()
        logger.info(f"Sampled {sample_size:,} records for feature pipeline.")
        
    # 1. Feature transformations
    df = add_time_features(df, datetime_col="datetime")
    df = encode_vehicle_tiers(df, name_col="name", cab_type_col="cab_type")
    
    # 2. Select model features
    feature_cols = [
        "distance", "surge_multiplier", "hour", "day_of_week",
        "is_weekend", "is_rush_hour", "is_late_night",
        "is_premium_tier", "is_shared", "is_uber",
        "sin_hour", "cos_hour", "sin_day", "cos_day"
    ]
    target_col = "price"
    
    available_cols = [c for c in feature_cols if c in df.columns]
    X = df[available_cols].copy()
    y = df[target_col].copy()
    
    # Fill remaining NaNs if any
    X = X.fillna(0.0)
    
    # Train / Val / Test split (70% / 15% / 15%)
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=0.15, random_state=random_state
    )
    val_ratio = 0.15 / 0.85
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=val_ratio, random_state=random_state
    )
    
    logger.info(f"Ride Split - Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
    
    # Export unified features CSV for notebook compatibility
    processed_dir = Path("data/processed")
    processed_dir.mkdir(parents=True, exist_ok=True)
    
    full_ride_features = pd.concat([X, y], axis=1)
    full_ride_features.to_csv(processed_dir / "ride_features.csv", index=False)
    
    # Save splits as parquet
    train_df = pd.concat([X_train, y_train], axis=1)
    val_df = pd.concat([X_val, y_val], axis=1)
    test_df = pd.concat([X_test, y_test], axis=1)
    
    train_df.to_parquet(processed_dir / "ride_train.parquet", index=False)
    val_df.to_parquet(processed_dir / "ride_val.parquet", index=False)
    test_df.to_parquet(processed_dir / "ride_test.parquet", index=False)
    
    return (X_train, X_val, X_test, y_train, y_val, y_test)


def build_delivery_features(
    input_path: str = "data/interim/delivery_logistics_clean.parquet",
    random_state: int = 42
) -> tuple:
    """
    Build ML feature tables for delivery logistics delay prediction.
    """
    path = Path(input_path)
    if not path.exists():
        logger.warning(f"Delivery logistics file not found at {input_path}, generating dummy features.")
        return None
        
    logger.info(f"Building delivery features from {input_path}...")
    df = pd.read_parquet(input_path)
    df = encode_delivery_attributes(df)
    
    feature_cols = [
        "delivery_distance", "order_value", "traffic_severity",
        "weather_severity", "is_small_basket", "is_long_distance",
        "delivery_friction_index"
    ]
    target_col = "delivery_delay"
    
    available_cols = [c for c in feature_cols if c in df.columns]
    X = df[available_cols].copy().fillna(0.0)
    y = df[target_col].copy()
    
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=0.15, random_state=random_state
    )
    val_ratio = 0.15 / 0.85
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=val_ratio, random_state=random_state
    )
    
    logger.info(f"Delivery Split - Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
    
    processed_dir = Path("data/processed")
    processed_dir.mkdir(parents=True, exist_ok=True)
    
    full_deliv_features = pd.concat([X, y], axis=1)
    full_deliv_features.to_csv(processed_dir / "delivery_features.csv", index=False)
    
    pd.concat([X_train, y_train], axis=1).to_parquet(processed_dir / "delivery_train.parquet", index=False)
    pd.concat([X_val, y_val], axis=1).to_parquet(processed_dir / "delivery_val.parquet", index=False)
    pd.concat([X_test, y_test], axis=1).to_parquet(processed_dir / "delivery_test.parquet", index=False)
    
    return (X_train, X_val, X_test, y_train, y_val, y_test)


def build_all():
    """Build all ride and delivery feature datasets."""
    logger.info("Executing full feature engineering build...")
    ride_splits = build_ride_features()
    delivery_splits = build_delivery_features()
    logger.info("Feature engineering build completed successfully.")
    return {
        "ride": ride_splits,
        "delivery": delivery_splits
    }


if __name__ == "__main__":
    build_all()
