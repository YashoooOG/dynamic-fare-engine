"""
Unit tests for Feature Engineering Modules (time_features, zone_encoding, lag_features).
"""

import unittest
import pandas as pd
import numpy as np

from src.features.time_features import add_time_features
from src.features.zone_encoding import encode_vehicle_tiers, encode_delivery_attributes, haversine_distance
from src.features.lag_features import add_rolling_zone_features, add_lag_features


class TestFeatureEngineering(unittest.TestCase):

    def test_add_time_features(self):
        df = pd.DataFrame({
            "datetime": pd.date_range("2026-08-01 08:00:00", periods=5, freq="h")
        })
        res = add_time_features(df, datetime_col="datetime")
        
        self.assertIn("hour", res.columns)
        self.assertIn("day_of_week", res.columns)
        self.assertIn("is_rush_hour", res.columns)
        self.assertIn("sin_hour", res.columns)
        self.assertIn("cos_hour", res.columns)
        self.assertEqual(res.loc[0, "hour"], 8)
        self.assertEqual(res.loc[0, "is_rush_hour"], 1)

    def test_encode_vehicle_tiers(self):
        df = pd.DataFrame({
            "name": ["UberX", "Black", "UberPool", "Lyft Lux"],
            "cab_type": ["Uber", "Uber", "Uber", "Lyft"]
        })
        res = encode_vehicle_tiers(df)
        
        self.assertEqual(res.loc[0, "is_premium_tier"], 0)
        self.assertEqual(res.loc[1, "is_premium_tier"], 1)
        self.assertEqual(res.loc[2, "is_shared"], 1)
        self.assertEqual(res.loc[3, "is_premium_tier"], 1)
        self.assertEqual(res.loc[3, "is_uber"], 0)

    def test_encode_delivery_attributes(self):
        df = pd.DataFrame({
            "traffic_condition": ["Low", "Jam"],
            "weather_condition": ["Clear", "Rainy"],
            "order_value": [150.0, 500.0],
            "delivery_distance": [3.0, 12.0]
        })
        res = encode_delivery_attributes(df)
        
        self.assertIn("delivery_friction_index", res.columns)
        self.assertEqual(res.loc[0, "is_small_basket"], 1)
        self.assertEqual(res.loc[1, "is_small_basket"], 0)
        self.assertEqual(res.loc[0, "is_long_distance"], 0)
        self.assertEqual(res.loc[1, "is_long_distance"], 1)
        self.assertGreater(res.loc[1, "delivery_friction_index"], res.loc[0, "delivery_friction_index"])

    def test_haversine_distance(self):
        # Distance between Boston Common and Fenway Park (~2 miles)
        dist = haversine_distance(-71.066, 42.355, -71.097, 42.346)
        self.assertTrue(1.5 <= dist <= 2.5)

    def test_lag_features(self):
        df = pd.DataFrame({
            "zone": ["A"] * 6,
            "hour": [1, 2, 3, 4, 5, 6],
            "rides": [10, 15, 20, 25, 30, 35]
        })
        res = add_lag_features(df, group_col="zone", time_col="hour", value_col="rides", lags=[1], windows=[2])
        self.assertIn("rides_lag_1", res.columns)
        self.assertIn("rides_rolling_mean_2", res.columns)
        self.assertEqual(res.loc[1, "rides_lag_1"], 10)


if __name__ == "__main__":
    unittest.main()
