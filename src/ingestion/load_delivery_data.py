"""
Ingestion module for Food Delivery & Demand Data.
Reads raw food demand, center info, meal metadata, and logistics data,
performs joins, cleans nulls/duplicates, derives discounts and metrics,
and exports clean interim parquet datasets.
"""

import os
import sys
import logging
from pathlib import Path
import pandas as pd
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def load_raw_demand_data(
    train_path: str = "data/raw/delivery/food_demand_train.csv",
    centers_path: str = "data/raw/delivery/fulfilment_center_info.csv",
    meals_path: str = "data/raw/delivery/meal_info.csv"
) -> pd.DataFrame:
    """Load and merge raw food demand tables (train, centers, meals)."""
    for p in [train_path, centers_path, meals_path]:
        if not Path(p).exists():
            raise FileNotFoundError(f"Required raw file not found at: {p}")

    logger.info("Loading food demand tables...")
    train_df = pd.read_csv(train_path)
    centers_df = pd.read_csv(centers_path)
    meals_df = pd.read_csv(meals_path)

    logger.info(f"Loaded {len(train_df):,} demand rows, {len(centers_df):,} centers, {len(meals_df):,} meals.")
    
    # Merge tables
    merged_df = train_df.merge(centers_df, on="center_id", how="left").merge(meals_df, on="meal_id", how="left")
    return merged_df


def clean_demand_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean merged weekly food demand data:
    - Drop duplicate (week, center_id, meal_id) rows
    - Compute discount amounts and discount percentages
    - Handle null values
    - Add promotion flags
    """
    logger.info("Cleaning weekly food demand dataset...")
    clean_df = df.copy()
    
    # Drop duplicates
    if {"week", "center_id", "meal_id"}.issubset(clean_df.columns):
        dups = clean_df.duplicated(subset=["week", "center_id", "meal_id"]).sum()
        if dups > 0:
            clean_df = clean_df.drop_duplicates(subset=["week", "center_id", "meal_id"])
            logger.info(f"Dropped {dups:,} duplicate records.")

    # Calculate discount metrics
    if "base_price" in clean_df.columns and "checkout_price" in clean_df.columns:
        clean_df["discount_amount"] = clean_df["base_price"] - clean_df["checkout_price"]
        clean_df["discount_pct"] = (
            (clean_df["discount_amount"] / clean_df["base_price"].replace(0, np.nan)) * 100
        ).clip(lower=0.0).fillna(0.0)

    # Impute missing categorical values
    for col in ["center_type", "category", "cuisine"]:
        if col in clean_df.columns:
            clean_df[col] = clean_df[col].fillna("Unknown")

    logger.info(f"Cleaned demand data shape: {clean_df.shape}")
    return clean_df


def load_and_clean_logistics_data(
    logistics_path: str = "data/raw/delivery/food_delivery_india.csv"
) -> pd.DataFrame:
    """Load and clean real-world operational food delivery logistics data."""
    path = Path(logistics_path)
    if not path.exists():
        logger.warning(f"Logistics file not found at {logistics_path}, skipping.")
        return pd.DataFrame()

    logger.info(f"Loading logistics dataset from {logistics_path}...")
    df = pd.read_csv(logistics_path)
    
    # Clean delivery delays and order values
    clean_df = df.dropna(subset=["delivery_delay", "order_value", "delivery_distance"]).copy()
    
    # Standardize traffic and weather condition values
    clean_df["traffic_condition"] = clean_df["traffic_condition"].fillna("Medium").astype(str).str.strip()
    clean_df["weather_condition"] = clean_df["weather_condition"].fillna("Clear").astype(str).str.strip()
    
    # Map traffic and weather severity
    traffic_map = {"Low": 1, "Medium": 2, "High": 3, "Jam": 4}
    weather_map = {
        "Clear": 1, "Sunny": 1, "Windy": 2, "Fog": 2,
        "Cloudy": 2, "Sandstorms": 3, "Stormy": 4, "Rainy": 4
    }
    clean_df["traffic_severity"] = clean_df["traffic_condition"].map(traffic_map).fillna(2).astype(int)
    clean_df["weather_severity"] = clean_df["weather_condition"].map(weather_map).fillna(2).astype(int)
    
    # Composite friction index
    clean_df["delivery_friction_index"] = (clean_df["traffic_severity"] * 0.5) + (clean_df["weather_severity"] * 0.5)
    clean_df["is_small_basket"] = (clean_df["order_value"] < 300).astype(int)
    clean_df["is_long_distance"] = (clean_df["delivery_distance"] > 10.0).astype(int)

    logger.info(f"Cleaned logistics records: {len(clean_df):,}")
    return clean_df


def save_interim_delivery_data(
    demand_df: pd.DataFrame,
    logistics_df: pd.DataFrame,
    demand_out_path: str = "data/interim/delivery_weekly_clean.parquet",
    logistics_out_path: str = "data/interim/delivery_logistics_clean.parquet"
) -> None:
    """Save cleaned delivery datasets to interim parquet format."""
    Path(demand_out_path).parent.mkdir(parents=True, exist_ok=True)
    
    if not demand_df.empty:
        demand_df.to_parquet(demand_out_path, index=False, engine="pyarrow")
        logger.info(f"Saved weekly demand data to {demand_out_path} ({len(demand_df):,} rows)")
        
    if not logistics_df.empty:
        logistics_df.to_parquet(logistics_out_path, index=False, engine="pyarrow")
        logger.info(f"Saved logistics data to {logistics_out_path} ({len(logistics_df):,} rows)")


def run_ingestion(
    train_path: str = "data/raw/delivery/food_demand_train.csv",
    centers_path: str = "data/raw/delivery/fulfilment_center_info.csv",
    meals_path: str = "data/raw/delivery/meal_info.csv",
    logistics_path: str = "data/raw/delivery/food_delivery_india.csv",
    demand_out: str = "data/interim/delivery_weekly_clean.parquet",
    logistics_out: str = "data/interim/delivery_logistics_clean.parquet"
) -> tuple:
    """Full execution pipeline for delivery data ingestion."""
    raw_demand = load_raw_demand_data(train_path, centers_path, meals_path)
    clean_demand = clean_demand_data(raw_demand)
    clean_logistics = load_and_clean_logistics_data(logistics_path)
    save_interim_delivery_data(clean_demand, clean_logistics, demand_out, logistics_out)
    return clean_demand, clean_logistics


if __name__ == "__main__":
    run_ingestion()
