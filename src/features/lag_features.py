"""
Feature engineering module for Lagged, Rolling, and Historical Aggregations.
"""

import pandas as pd
import numpy as np


def add_rolling_zone_features(
    df: pd.DataFrame,
    zone_col: str = "source",
    time_col: str = "hour",
    price_col: str = "price"
) -> pd.DataFrame:
    """
    Compute zone-level historical demand velocity and mean fare aggregates.
    """
    out = df.copy()
    if zone_col in out.columns and price_col in out.columns:
        zone_stats = out.groupby(zone_col)[price_col].agg(["mean", "std", "count"]).reset_index()
        zone_stats.columns = [zone_col, f"{zone_col}_mean_fare", f"{zone_col}_std_fare", f"{zone_col}_ride_vol"]
        zone_stats[f"{zone_col}_std_fare"] = zone_stats[f"{zone_col}_std_fare"].fillna(0.0)
        out = out.merge(zone_stats, on=zone_col, how="left")
        
    return out


def add_lag_features(
    df: pd.DataFrame,
    group_col: str,
    time_col: str,
    value_col: str,
    lags: list = (1, 24),
    windows: list = (3, 6)
) -> pd.DataFrame:
    """
    Generate time-lagged and rolling-window statistical aggregates for panel data.
    """
    out = df.sort_values(by=[group_col, time_col]).copy()
    
    for lag in lags:
        out[f"{value_col}_lag_{lag}"] = out.groupby(group_col)[value_col].shift(lag)
        
    for w in windows:
        out[f"{value_col}_rolling_mean_{w}"] = (
            out.groupby(group_col)[value_col]
            .transform(lambda s: s.shift(1).rolling(window=w, min_periods=1).mean())
        )
        
    return out
