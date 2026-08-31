"""
Ingestion module for Ride-Hailing Data.
Reads raw Boston Uber/Lyft dataset, cleans timestamps and missing values,
and exports clean interim parquet data.
"""

import os
import sys
import logging
from pathlib import Path
import pandas as pd
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def load_raw_ride_data(file_path: str = "data/raw/ride/uber_lyft_boston.csv") -> pd.DataFrame:
    """Load raw ride-hailing CSV data from disk."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Raw ride data file not found at: {file_path}")
    
    logger.info(f"Loading raw ride dataset from {file_path}...")
    df = pd.read_csv(file_path)
    logger.info(f"Loaded {len(df):,} raw records with columns: {list(df.columns)}")
    return df


def clean_ride_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean raw ride data:
    - Drop rows where price is null (metered taxis or unpriced quotes)
    - Convert timestamp (ms or s) to datetime
    - Derive basic temporal fields: hour, day, month, day_name, day_of_week
    - Drop duplicate rows
    - Fill/standardize categorical columns
    """
    logger.info("Starting ride data cleaning pipeline...")
    initial_rows = len(df)
    
    # 1. Filter out missing target price
    clean_df = df.dropna(subset=["price"]).copy()
    dropped_nulls = initial_rows - len(clean_df)
    logger.info(f"Dropped {dropped_nulls:,} rows with missing price.")
    
    # 2. Timestamp conversion
    if "time_stamp" in clean_df.columns:
        unit = "ms" if clean_df["time_stamp"].max() > 1e11 else "s"
        clean_df["datetime"] = pd.to_datetime(clean_df["time_stamp"], unit=unit)
    elif "timestamp" in clean_df.columns:
        clean_df["datetime"] = pd.to_datetime(clean_df["timestamp"])
    elif "pickup_datetime" in clean_df.columns:
        clean_df["datetime"] = pd.to_datetime(clean_df["pickup_datetime"])
    else:
        clean_df["datetime"] = pd.Timestamp.now()
        
    clean_df["hour"] = clean_df["datetime"].dt.hour.astype(int)
    clean_df["day"] = clean_df["datetime"].dt.day.astype(int)
    clean_df["month"] = clean_df["datetime"].dt.month.astype(int)
    clean_df["day_of_week"] = clean_df["datetime"].dt.dayofweek.astype(int)
    clean_df["day_name"] = clean_df["datetime"].dt.day_name()
    
    # 3. Standardize surge multiplier and distance
    if "surge_multiplier" in clean_df.columns:
        clean_df["surge_multiplier"] = clean_df["surge_multiplier"].fillna(1.0).astype(float)
    else:
        clean_df["surge_multiplier"] = 1.0
        
    if "distance" in clean_df.columns:
        clean_df["distance"] = clean_df["distance"].astype(float)
        
    # 4. Remove duplicate ride IDs if id column exists
    if "id" in clean_df.columns:
        dups = clean_df.duplicated(subset=["id"]).sum()
        if dups > 0:
            clean_df = clean_df.drop_duplicates(subset=["id"])
            logger.info(f"Dropped {dups:,} duplicate ride records.")
            
    logger.info(f"Cleaning complete. Output records: {len(clean_df):,}")
    return clean_df


def save_interim_ride_data(df: pd.DataFrame, output_path: str = "data/interim/ride_hourly_clean.parquet") -> str:
    """Save cleaned dataframe to interim parquet format."""
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Saving interim ride dataset to {output_path}...")
    df.to_parquet(output_path, index=False, engine="pyarrow")
    logger.info(f"Successfully saved {len(df):,} records to {output_path}")
    return output_path


def run_ingestion(raw_path: str = "data/raw/ride/uber_lyft_boston.csv",
                  out_path: str = "data/interim/ride_hourly_clean.parquet") -> pd.DataFrame:
    """Full execution pipeline for ride ingestion."""
    raw_df = load_raw_ride_data(raw_path)
    clean_df = clean_ride_data(raw_df)
    save_interim_ride_data(clean_df, out_path)
    return clean_df


if __name__ == "__main__":
    raw_file = sys.argv[1] if len(sys.argv) > 1 else "data/raw/ride/uber_lyft_boston.csv"
    out_file = sys.argv[2] if len(sys.argv) > 2 else "data/interim/ride_hourly_clean.parquet"
    run_ingestion(raw_file, out_file)
