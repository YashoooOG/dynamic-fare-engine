"""
Pipeline smoke tests for Ingestion, Features, and Models using unittest.
"""

import unittest
from pathlib import Path
import pandas as pd
import numpy as np

from src.features.time_features import add_time_features
from src.features.zone_encoding import encode_vehicle_tiers, encode_delivery_attributes
from src.models.demand_forecast import DemandForecastModel
from src.models.cancellation_model import CancellationModel


class TestPipeline(unittest.TestCase):
    
    def test_time_features(self):
        df = pd.DataFrame({
            "timestamp": pd.date_range("2026-08-01", periods=10, freq="h")
        })
        transformed = add_time_features(df, datetime_col="timestamp")
        self.assertIn("hour", transformed.columns)
        self.assertIn("day_of_week", transformed.columns)
        self.assertIn("sin_hour", transformed.columns)
        self.assertIn("cos_hour", transformed.columns)
        self.assertEqual(len(transformed), 10)

    def test_zone_encoding(self):
        df = pd.DataFrame({
            "name": ["UberX", "Black", "UberPool"],
            "cab_type": ["Uber", "Uber", "Uber"]
        })
        encoded = encode_vehicle_tiers(df)
        self.assertIn("is_premium_tier", encoded.columns)
        self.assertIn("is_shared", encoded.columns)
        self.assertEqual(encoded.loc[1, "is_premium_tier"], 1)
        self.assertEqual(encoded.loc[2, "is_shared"], 1)

    def test_model_loading_and_inference(self):
        ride_model_path = "models/demand_model_ride.joblib"
        cancel_model_path = "models/cancellation_model.joblib"
        
        self.assertTrue(Path(ride_model_path).exists())
        self.assertTrue(Path(cancel_model_path).exists())
        
        ride_model = DemandForecastModel.load(ride_model_path)
        cancel_model = CancellationModel.load(cancel_model_path)
        
        sample_ride = pd.DataFrame([{
            "distance": 3.5,
            "surge_multiplier": 1.2,
            "hour": 18,
            "day_of_week": 4,
            "is_weekend": 0,
            "is_rush_hour": 1,
            "is_late_night": 0,
            "is_premium_tier": 0,
            "is_shared": 0,
            "is_uber": 1,
            "sin_hour": np.sin(2 * np.pi * 18 / 24),
            "cos_hour": np.cos(2 * np.pi * 18 / 24),
            "sin_day": np.sin(2 * np.pi * 4 / 7),
            "cos_day": np.cos(2 * np.pi * 4 / 7)
        }])
        
        pred_fare = ride_model.predict(sample_ride)
        self.assertEqual(len(pred_fare), 1)
        self.assertGreater(pred_fare[0], 0)
        
        sample_cancel = pd.DataFrame([{
            "surge_multiplier": 1.5,
            "distance": 4.0,
            "fare": 18.5,
            "wait_time_min": 6.0,
            "traffic_severity": 2,
            "weather_severity": 1,
            "is_new_user": 0
        }])
        
        pred_risk = cancel_model.predict_proba(sample_cancel)
        self.assertEqual(len(pred_risk), 1)
        self.assertTrue(0.0 <= pred_risk[0] <= 1.0)


if __name__ == "__main__":
    unittest.main()
