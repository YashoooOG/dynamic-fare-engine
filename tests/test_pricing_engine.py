"""
Unit tests for Dynamic Pricing & Optimization Engine (surge_engine, optimizer).
"""

import unittest
import pandas as pd
import numpy as np

from src.pricing.surge_engine import DynamicFareEngine
from src.pricing.optimizer import PriceOptimizer


class TestPricingEngine(unittest.TestCase):

    def setUp(self):
        self.engine = DynamicFareEngine()
        self.optimizer = PriceOptimizer(self.engine)

    def test_surge_multiplier_calculation(self):
        # Balanced or excess supply -> 1.0x
        surge_normal = self.engine.calculate_surge_multiplier(demand=30, supply=40)
        self.assertEqual(surge_normal, 1.0)

        # High demand ratio -> surge > 1.0x
        surge_high = self.engine.calculate_surge_multiplier(demand=90, supply=30)
        self.assertGreater(surge_high, 1.5)
        self.assertLessEqual(surge_high, self.engine.max_surge)

        # Zero supply -> max surge
        surge_zero = self.engine.calculate_surge_multiplier(demand=50, supply=0)
        self.assertEqual(surge_zero, self.engine.max_surge)

    def test_ride_fare_calculation(self):
        # 5 miles, standard tier, surge = 1.0x
        quote = self.engine.calculate_ride_fare(
            distance_miles=5.0,
            duration_min=15.0,
            surge_multiplier=1.0,
            is_premium=False
        )
        self.assertEqual(quote["service_type"], "ride")
        self.assertGreater(quote["total_fare"], 0)
        self.assertEqual(quote["surge_multiplier"], 1.0)
        self.assertAlmostEqual(quote["driver_payout"] + quote["platform_fee"], quote["total_fare"], places=1)

        # Premium tier multiplier check
        premium_quote = self.engine.calculate_ride_fare(
            distance_miles=5.0,
            duration_min=15.0,
            surge_multiplier=1.0,
            is_premium=True
        )
        self.assertGreater(premium_quote["total_fare"], quote["total_fare"])

    def test_delivery_fee_calculation(self):
        # Standard delivery
        quote_norm = self.engine.calculate_delivery_fee(distance_km=3.0, order_value=400.0)
        self.assertGreater(quote_norm["total_delivery_fee"], 0)
        self.assertEqual(quote_norm["small_basket_surcharge"], 0.0)

        # Small basket surcharge (<250 INR)
        quote_small = self.engine.calculate_delivery_fee(distance_km=3.0, order_value=180.0)
        self.assertEqual(quote_small["small_basket_surcharge"], self.engine.small_basket_surcharge)
        self.assertGreater(quote_small["total_delivery_fee"], quote_norm["total_delivery_fee"])

    def test_optimizer_ride_surge(self):
        opt = self.optimizer.optimize_ride_surge(
            demand=80,
            supply=30,
            distance_miles=4.0
        )
        self.assertIn("optimal_surge", opt)
        self.assertIn("optimal_expected_revenue", opt)
        self.assertTrue(1.0 <= opt["optimal_surge"] <= self.engine.max_surge)
        self.assertFalse(opt["curve"].empty)

    def test_marketplace_simulation(self):
        sim = self.optimizer.run_marketplace_simulation(n_samples=50, seed=42)
        summary = sim["summary"]
        self.assertEqual(summary["total_requests"], 50)
        self.assertIn("dynamic_revenue", summary)
        self.assertIn("static_revenue", summary)
        self.assertEqual(len(sim["sim_df"]), 50)


if __name__ == "__main__":
    unittest.main()
