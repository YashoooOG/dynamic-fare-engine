"""
Feature engineering module for Spatial, Categorical, and Vehicle Tier Encodings.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder

PREMIUM_VEHICLE_NAMES = [
    "Lux", "Lux Black", "Lux Black XL", "Black", "Black SUV", "UberXL", "Executive"
]
SHARED_VEHICLE_NAMES = ["Shared", "Line", "Pool", "UberPool"]


def haversine_distance(lon1, lat1, lon2, lat2) -> np.ndarray:
    """Vectorized Haversine great-circle distance between coordinate pairs in miles."""
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0)**2
    c = 2 * np.arcsin(np.sqrt(a))
    miles = 6367 * c * 0.621371
    return miles


def encode_vehicle_tiers(df: pd.DataFrame, name_col: str = "name", cab_type_col: str = "cab_type") -> pd.DataFrame:
    """
    Derive categorical and binary service flags:
    - is_premium_tier (1 for Black/Lux/SUV tiers, else 0)
    - is_shared (1 for Pool/Shared, else 0)
    - is_uber (1 for Uber, 0 for Lyft)
    """
    out = df.copy()
    
    if name_col in out.columns:
        out["is_premium_tier"] = out[name_col].isin(PREMIUM_VEHICLE_NAMES).astype(int)
        out["is_shared"] = out[name_col].astype(str).str.contains("Shared|Line|Pool", case=False, na=False).astype(int)
    else:
        out["is_premium_tier"] = 0
        out["is_shared"] = 0
        
    if cab_type_col in out.columns:
        out["is_uber"] = (out[cab_type_col].astype(str).str.lower() == "uber").astype(int)
    else:
        out["is_uber"] = 1
        
    return out


def encode_delivery_attributes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encode delivery traffic, weather, and basket categories.
    """
    out = df.copy()
    
    traffic_map = {"Low": 1, "Medium": 2, "High": 3, "Jam": 4}
    weather_map = {
        "Clear": 1, "Sunny": 1, "Windy": 2, "Fog": 2,
        "Cloudy": 2, "Sandstorms": 3, "Stormy": 4, "Rainy": 4
    }
    
    if "traffic_condition" in out.columns and "traffic_severity" not in out.columns:
        out["traffic_severity"] = out["traffic_condition"].map(traffic_map).fillna(2).astype(int)
    if "weather_condition" in out.columns and "weather_severity" not in out.columns:
        out["weather_severity"] = out["weather_condition"].map(weather_map).fillna(2).astype(int)
        
    if "delivery_friction_index" not in out.columns:
        t_sev = out["traffic_severity"] if "traffic_severity" in out.columns else 2
        w_sev = out["weather_severity"] if "weather_severity" in out.columns else 2
        out["delivery_friction_index"] = (t_sev * 0.5) + (w_sev * 0.5)
        
    if "order_value" in out.columns and "is_small_basket" not in out.columns:
        out["is_small_basket"] = (out["order_value"] < 300).astype(int)
        
    if "delivery_distance" in out.columns and "is_long_distance" not in out.columns:
        out["is_long_distance"] = (out["delivery_distance"] > 10.0).astype(int)
        
    return out
