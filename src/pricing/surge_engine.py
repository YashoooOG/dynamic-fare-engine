"""
Dynamic Surge Pricing Engine Module.
Implements multi-modal real-time surge pricing algorithms for Ride-Hailing and Food Delivery,
integrating trained XGBoost demand & cancellation models with economic price elasticity curves.
"""

import os
import logging
from pathlib import Path
import numpy as np
import pandas as pd
import yaml

from src.models.demand_forecast import DemandForecastModel
from src.models.cancellation_model import CancellationModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class DynamicFareEngine:
    """
    Core Dynamic Pricing Engine supporting:
    - Ride-Hailing Dynamic Surge Pricing & Tier Economics
    - On-Demand Food Delivery Dynamic Fees with Weather/Traffic Friction
    - Microeconomic Price Elasticity & Cancellation Risk Caps
    - ML-guided Demand and Delay Forecasting
    """

    def __init__(self, config_path: str = "config.yaml"):
        self.config = self._load_config(config_path)
        p_cfg = self.config.get("pricing", {})
        
        # Ride parameters
        self.base_fare = float(p_cfg.get("base_fare", 3.50))
        self.per_mile_rate = float(p_cfg.get("per_mile_rate", 1.85))
        self.per_minute_rate = float(p_cfg.get("per_minute_rate", 0.35))
        self.premium_multiplier = float(p_cfg.get("premium_multiplier", 1.60))
        self.shared_discount = float(p_cfg.get("shared_discount", 0.80))
        self.min_surge = float(p_cfg.get("min_surge", 1.0))
        self.max_surge = float(p_cfg.get("max_surge", 3.5))
        self.surge_step = float(p_cfg.get("surge_step", 0.1))
        self.cancel_risk_cap = float(p_cfg.get("cancel_risk_cap", 0.45))
        
        # Delivery parameters
        self.base_delivery_fee = float(p_cfg.get("base_delivery_fee", 30.0))
        self.per_km_delivery_rate = float(p_cfg.get("per_km_delivery_rate", 8.0))
        self.small_basket_threshold = float(p_cfg.get("small_basket_threshold", 250.0))
        self.small_basket_surcharge = float(p_cfg.get("small_basket_surcharge", 15.0))
        
        # ML Models
        self.ride_demand_model = None
        self.delivery_delay_model = None
        self.cancellation_model = None
        self._load_ml_models()

    def _load_config(self, config_path: str) -> dict:
        path = Path(config_path)
        if path.exists():
            try:
                with open(path, "r") as f:
                    return yaml.safe_load(f) or {}
            except Exception as e:
                logger.warning(f"Failed to read {config_path}: {e}")
        return {}

    def _load_ml_models(self):
        """Lazily load trained machine learning models if available."""
        models_dir = Path(self.config.get("paths", {}).get("models_dir", "models"))
        
        # 1. Ride Demand Model
        ride_path = models_dir / "demand_model_ride.joblib"
        if ride_path.exists():
            try:
                self.ride_demand_model = DemandForecastModel.load(str(ride_path))
            except Exception as e:
                logger.warning(f"Could not load ride demand model: {e}")

        # 2. Delivery Delay / Friction Model
        deliv_path = models_dir / "demand_model_delivery.joblib"
        if deliv_path.exists():
            try:
                self.delivery_delay_model = DemandForecastModel.load(str(deliv_path))
            except Exception as e:
                logger.warning(f"Could not load delivery model: {e}")

        # 3. Cancellation Risk Classifier
        cancel_path = models_dir / "cancellation_model.joblib"
        if cancel_path.exists():
            try:
                self.cancellation_model = CancellationModel.load(str(cancel_path))
            except Exception as e:
                logger.warning(f"Could not load cancellation model: {e}")

    def calculate_surge_multiplier(
        self,
        demand: float,
        supply: float,
        sensitivity: float = 0.85,
        zone_multiplier: float = 1.0
    ) -> float:
        """
        Compute real-time surge multiplier using smooth logistic supply-demand elasticity.
        
        Formula:
        Ratio = Demand / max(1, Supply)
        If Ratio <= 1.0: Surge = 1.0
        If Ratio > 1.0: Surge = 1.0 + Sensitivity * (Ratio - 1.0) * ZoneMultiplier
        Clamped within [min_surge, max_surge] and rounded to surge_step.
        """
        if supply <= 0:
            return round(self.max_surge, 2)
            
        ratio = float(demand) / max(1.0, float(supply))
        if ratio <= 1.0:
            raw_surge = self.min_surge
        else:
            raw_surge = 1.0 + sensitivity * (ratio - 1.0) * zone_multiplier

        clipped = float(np.clip(raw_surge, self.min_surge, self.max_surge))
        # Round to step
        stepped = round(round(clipped / self.surge_step) * self.surge_step, 2)
        return float(np.clip(stepped, self.min_surge, self.max_surge))

    def estimate_cancellation_risk(
        self,
        surge_multiplier: float,
        distance_miles: float,
        fare: float,
        wait_time_min: float = 5.0,
        traffic_severity: int = 1,
        weather_severity: int = 1,
        is_new_user: int = 0
    ) -> float:
        """
        Predict probability of customer rejection or cancellation.
        Uses trained Cancellation XGBoost model if available, fallback to logistic elasticity.
        """
        if self.cancellation_model and self.cancellation_model.is_fitted:
            sample = pd.DataFrame([{
                "surge_multiplier": float(surge_multiplier),
                "distance": float(distance_miles),
                "fare": float(fare),
                "wait_time_min": float(wait_time_min),
                "traffic_severity": int(traffic_severity),
                "weather_severity": int(weather_severity),
                "is_new_user": int(is_new_user)
            }])
            try:
                prob = self.cancellation_model.predict_proba(sample)[0]
                return round(float(np.clip(prob, 0.02, 0.98)), 3)
            except Exception as e:
                logger.debug(f"Model prediction fallback: {e}")

        # Mathematical fallback
        z = (
            -2.50
            + 1.35 * (surge_multiplier - 1.0)
            + 0.09 * wait_time_min
            + 0.12 * (traffic_severity - 1)
            + 0.15 * (weather_severity - 1)
            + 0.30 * is_new_user
        )
        prob = 1.0 / (1.0 + np.exp(-z))
        return round(float(np.clip(prob, 0.02, 0.98)), 3)

    def calculate_ride_fare(
        self,
        distance_miles: float,
        duration_min: float = None,
        surge_multiplier: float = 1.0,
        is_premium: bool = False,
        is_shared: bool = False,
        wait_time_min: float = 5.0,
        traffic_severity: int = 1,
        weather_severity: int = 1,
        is_new_user: int = 0,
        hour: int = 14,
        day_of_week: int = 2
    ) -> dict:
        """
        Calculate complete ride-hailing quote and granular price breakdown.
        """
        dist = max(0.2, float(distance_miles))
        if duration_min is None:
            # Estimate 3.2 mins per mile + traffic delay
            duration_min = dist * 3.2 + (traffic_severity - 1) * 3.0 + 2.0
        dur = max(1.0, float(duration_min))

        surge = float(np.clip(surge_multiplier, self.min_surge, self.max_surge))
        
        # Base Fare Components
        dist_cost = dist * self.per_mile_rate
        time_cost = dur * self.per_minute_rate
        subtotal = self.base_fare + dist_cost + time_cost

        # Vehicle Tier Multiplier
        tier_multiplier = 1.0
        tier_label = "Standard (UberX/Lyft)"
        if is_premium:
            tier_multiplier = self.premium_multiplier
            tier_label = "Premium (Black/Lux)"
        elif is_shared:
            tier_multiplier = self.shared_discount
            tier_label = "Shared (UberPool)"

        tiered_base = subtotal * tier_multiplier
        total_fare = round(tiered_base * surge, 2)
        
        # ML Predicted baseline reference if available
        ml_predicted_price = None
        if self.ride_demand_model and self.ride_demand_model.is_fitted:
            try:
                features = pd.DataFrame([{
                    "distance": dist,
                    "surge_multiplier": surge,
                    "hour": hour,
                    "day_of_week": day_of_week,
                    "is_weekend": int(day_of_week in [5, 6]),
                    "is_rush_hour": int(hour in [7, 8, 9, 16, 17, 18, 19]),
                    "is_late_night": int(hour in [23, 0, 1, 2, 3, 4]),
                    "is_premium_tier": int(is_premium),
                    "is_shared": int(is_shared),
                    "is_uber": 1,
                    "sin_hour": np.sin(2 * np.pi * hour / 24.0),
                    "cos_hour": np.cos(2 * np.pi * hour / 24.0),
                    "sin_day": np.sin(2 * np.pi * day_of_week / 7.0),
                    "cos_day": np.cos(2 * np.pi * day_of_week / 7.0)
                }])
                ml_predicted_price = round(float(self.ride_demand_model.predict(features)[0]), 2)
            except Exception as e:
                logger.debug(f"ML reference inference skipped: {e}")

        # Cancellation risk
        cancel_risk = self.estimate_cancellation_risk(
            surge_multiplier=surge,
            distance_miles=dist,
            fare=total_fare,
            wait_time_min=wait_time_min,
            traffic_severity=traffic_severity,
            weather_severity=weather_severity,
            is_new_user=is_new_user
        )

        driver_payout = round(total_fare * 0.78, 2)
        platform_fee = round(total_fare - driver_payout, 2)

        return {
            "service_type": "ride",
            "tier_label": tier_label,
            "distance_miles": round(dist, 2),
            "duration_min": round(dur, 1),
            "base_fare": round(self.base_fare, 2),
            "distance_cost": round(dist_cost, 2),
            "time_cost": round(time_cost, 2),
            "tier_multiplier": tier_multiplier,
            "surge_multiplier": surge,
            "total_fare": total_fare,
            "driver_payout": driver_payout,
            "platform_fee": platform_fee,
            "cancellation_risk": cancel_risk,
            "ml_predicted_fare": ml_predicted_price or total_fare
        }

    def calculate_delivery_fee(
        self,
        distance_km: float,
        order_value: float,
        surge_multiplier: float = 1.0,
        weather_severity: int = 1,
        traffic_severity: int = 1
    ) -> dict:
        """
        Calculate dynamic food delivery fee with environmental and operational friction.
        """
        dist = max(0.5, float(distance_km))
        val = max(10.0, float(order_value))
        surge = float(np.clip(surge_multiplier, self.min_surge, self.max_surge))

        # Base delivery distance formula
        extra_dist = max(0.0, dist - 2.0)
        dist_fee = extra_dist * self.per_km_delivery_rate
        base_subtotal = self.base_delivery_fee + dist_fee
        surged_base = base_subtotal * surge

        # Surcharges
        small_basket = self.small_basket_surcharge if val < self.small_basket_threshold else 0.0
        weather_fee = (weather_severity - 1) * 10.0
        traffic_fee = (traffic_severity - 1) * 5.0
        
        # Delay forecast
        predicted_delay_min = round(dist * 2.2 + weather_fee * 1.5 + traffic_fee * 2.0, 1)
        if self.delivery_delay_model and self.delivery_delay_model.is_fitted:
            try:
                friction_idx = (traffic_severity * 0.5) + (weather_severity * 0.5)
                features = pd.DataFrame([{
                    "delivery_distance": dist,
                    "order_value": val,
                    "traffic_severity": traffic_severity,
                    "weather_severity": weather_severity,
                    "is_small_basket": int(val < 300),
                    "is_long_distance": int(dist > 10.0),
                    "delivery_friction_index": friction_idx
                }])
                predicted_delay_min = round(float(self.delivery_delay_model.predict(features)[0]), 1)
            except Exception as e:
                logger.debug(f"Delivery ML delay skipped: {e}")

        total_delivery_fee = round(surged_base + small_basket + weather_fee + traffic_fee, 2)
        driver_payout = round(total_delivery_fee * 0.80, 2)
        platform_fee = round(total_delivery_fee - driver_payout, 2)

        return {
            "service_type": "delivery",
            "distance_km": round(dist, 2),
            "order_value": round(val, 2),
            "base_fee": round(self.base_delivery_fee, 2),
            "distance_fee": round(dist_fee, 2),
            "surge_multiplier": surge,
            "small_basket_surcharge": round(small_basket, 2),
            "weather_surcharge": round(weather_fee, 2),
            "traffic_surcharge": round(traffic_fee, 2),
            "predicted_delay_min": max(10.0, predicted_delay_min),
            "total_delivery_fee": total_delivery_fee,
            "driver_payout": driver_payout,
            "platform_fee": platform_fee
        }


# Alias for backward compatibility
SurgePricingEngine = DynamicFareEngine

if __name__ == "__main__":
    engine = DynamicFareEngine()
    print("Testing ride fare calculation:")
    r = engine.calculate_ride_fare(distance_miles=4.5, surge_multiplier=1.4, is_premium=True)
    print(r)
    print("\nTesting delivery fee calculation:")
    d = engine.calculate_delivery_fee(distance_km=5.0, order_value=200, surge_multiplier=1.2, weather_severity=2)
    print(d)
