"""
Feature engineering module for Temporal and Cyclical transformations.
"""

import numpy as np
import pandas as pd


def add_time_features(df: pd.DataFrame, datetime_col: str = "datetime") -> pd.DataFrame:
    """
    Extract calendar, diurnal, and cyclical time features from datetime column.
    
    Derived columns:
    - hour (0-23)
    - day_of_week (0-6, where 0=Monday)
    - day (1-31)
    - month (1-12)
    - is_weekend (0 or 1)
    - is_rush_hour (0 or 1: 7-9 AM, 16-19 PM)
    - is_late_night (0 or 1: 23-4 AM)
    - sin_hour, cos_hour (cyclical 24-hour periodic encoding)
    - sin_day, cos_day (cyclical 7-day periodic encoding)
    """
    out = df.copy()
    
    if datetime_col not in out.columns:
        if "time_stamp" in out.columns:
            unit = "ms" if out["time_stamp"].max() > 1e11 else "s"
            out[datetime_col] = pd.to_datetime(out["time_stamp"], unit=unit)
        elif "timestamp" in out.columns:
            out[datetime_col] = pd.to_datetime(out["timestamp"])
        elif "pickup_datetime" in out.columns:
            out[datetime_col] = pd.to_datetime(out["pickup_datetime"])
        else:
            out[datetime_col] = pd.Timestamp.now()

    dt = out[datetime_col]
    out["hour"] = dt.dt.hour.astype(int)
    out["day_of_week"] = dt.dt.dayofweek.astype(int)
    out["day"] = dt.dt.day.astype(int)
    out["month"] = dt.dt.month.astype(int)
    
    # Behavioral temporal flags
    out["is_weekend"] = out["day_of_week"].isin([5, 6]).astype(int)
    out["is_rush_hour"] = out["hour"].isin([7, 8, 9, 16, 17, 18, 19]).astype(int)
    out["is_late_night"] = out["hour"].isin([23, 0, 1, 2, 3, 4]).astype(int)
    
    # Continuous cyclical transformations
    out["sin_hour"] = np.sin(2 * np.pi * out["hour"] / 24.0)
    out["cos_hour"] = np.cos(2 * np.pi * out["hour"] / 24.0)
    out["sin_day"] = np.sin(2 * np.pi * out["day_of_week"] / 7.0)
    out["cos_day"] = np.cos(2 * np.pi * out["day_of_week"] / 7.0)
    
    return out
