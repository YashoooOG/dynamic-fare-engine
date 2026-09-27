"""
Revenue Optimization & Monte Carlo Pricing Simulator Module.
Computes optimal revenue-maximizing surge multipliers subject to customer cancellation risk caps,
and executes microeconomic marketplace simulations comparing Static vs. Dynamic pricing strategies.
"""

import os
import logging
from pathlib import Path
import numpy as np
import pandas as pd

from src.pricing.surge_engine import DynamicFareEngine

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class PriceOptimizer:
    """
    Constrained Revenue Optimization Engine.
    Maximizes expected platform gross revenue E[R] while bounding customer drop-off risk.
    """

    def __init__(self, engine: DynamicFareEngine = None, cancel_risk_cap: float = 0.45):
        self.engine = engine or DynamicFareEngine()
        self.cancel_risk_cap = cancel_risk_cap

    def optimize_ride_surge(
        self,
        demand: float,
        supply: float,
        distance_miles: float,
        duration_min: float = None,
        is_premium: bool = False,
        traffic_severity: int = 1,
        weather_severity: int = 1,
        is_new_user: int = 0
    ) -> dict:
        """
        Find optimal surge multiplier s* that maximizes:
            E[Revenue(s)] = Fare(s) * P_accept(s) * P_fulfill(s)
        Subject to:
            P_cancel(s) <= cancel_risk_cap
            min_surge <= s <= max_surge
        """
        candidate_surges = np.arange(self.engine.min_surge, self.engine.max_surge + 0.05, 0.1)
        curve_data = []

        best_surge = 1.0
        max_expected_revenue = -1.0
        optimal_point = {}

        ratio = float(demand) / max(1.0, float(supply))

        for s in candidate_surges:
            s_val = round(float(s), 2)
            quote = self.engine.calculate_ride_fare(
                distance_miles=distance_miles,
                duration_min=duration_min,
                surge_multiplier=s_val,
                is_premium=is_premium,
                traffic_severity=traffic_severity,
                weather_severity=weather_severity,
                is_new_user=is_new_user
            )

            fare = quote["total_fare"]
            p_cancel = quote["cancellation_risk"]
            p_accept = max(0.02, 1.0 - p_cancel)

            # Driver supply response: higher surge incentivizes more drivers to enter the zone
            supply_boost = float(supply) * (1.0 + 0.35 * (s_val - 1.0))
            p_fulfill = float(np.clip(supply_boost / max(1.0, float(demand)), 0.05, 1.0))

            expected_revenue = fare * p_accept * p_fulfill
            is_valid = p_cancel <= self.cancel_risk_cap

            point = {
                "surge_multiplier": s_val,
                "fare": fare,
                "cancellation_risk": p_cancel,
                "acceptance_probability": round(p_accept, 3),
                "fulfillment_probability": round(p_fulfill, 3),
                "expected_revenue": round(expected_revenue, 2),
                "is_risk_feasible": is_valid
            }
            curve_data.append(point)

            if is_valid and expected_revenue > max_expected_revenue:
                max_expected_revenue = expected_revenue
                best_surge = s_val
                optimal_point = point

        # Fallback if no point was strictly under cap
        if not optimal_point and curve_data:
            optimal_point = min(curve_data, key=lambda x: x["cancellation_risk"])
            best_surge = optimal_point["surge_multiplier"]

        rule_based_surge = self.engine.calculate_surge_multiplier(demand=demand, supply=supply)

        return {
            "optimal_surge": best_surge,
            "rule_based_surge": rule_based_surge,
            "optimal_expected_revenue": optimal_point.get("expected_revenue", 0.0),
            "optimal_cancellation_risk": optimal_point.get("cancellation_risk", 0.0),
            "optimal_fare": optimal_point.get("fare", 0.0),
            "curve": pd.DataFrame(curve_data)
        }

    def run_marketplace_simulation(
        self,
        n_samples: int = 1500,
        seed: int = 42,
        price_sensitivity: float = 1.4
    ) -> dict:
        """
        Execute full Monte Carlo simulation benchmarking:
        1. Static Fixed Pricing (Surge = 1.0x)
        2. Dynamic Surge Pricing Engine
        """
        np.random.seed(seed)
        distances = np.random.exponential(scale=3.5, size=n_samples).clip(0.8, 15.0)
        durations = distances * np.random.uniform(2.5, 4.5, size=n_samples) + np.random.uniform(2, 8, size=n_samples)
        is_premium_list = np.random.choice([0, 1], size=n_samples, p=[0.8, 0.2])

        demands = np.random.randint(20, 95, size=n_samples)
        supplies = np.random.randint(18, 55, size=n_samples)

        simulation_records = []

        for i in range(n_samples):
            dist = float(distances[i])
            dur = float(durations[i])
            prem = bool(is_premium_list[i])
            dem = float(demands[i])
            sup = float(supplies[i])

            # 1. Static Pricing Strategy (Surge = 1.0x)
            static_quote = self.engine.calculate_ride_fare(
                distance_miles=dist,
                duration_min=dur,
                surge_multiplier=1.0,
                is_premium=prem
            )
            static_price = static_quote["total_fare"]
            static_accept_prob = max(0.05, 1.0 - static_quote["cancellation_risk"])
            static_accepted = np.random.rand() < static_accept_prob
            static_fulfilled = static_accepted and (np.random.rand() < min(1.0, sup / max(1.0, dem)))
            static_rev = static_price if static_fulfilled else 0.0

            # 2. Dynamic Surge Pricing Strategy
            surge = self.engine.calculate_surge_multiplier(demand=dem, supply=sup)
            dynamic_quote = self.engine.calculate_ride_fare(
                distance_miles=dist,
                duration_min=dur,
                surge_multiplier=surge,
                is_premium=prem
            )
            dynamic_price = dynamic_quote["total_fare"]
            dynamic_accept_prob = max(0.05, 1.0 - dynamic_quote["cancellation_risk"])
            dynamic_accepted = np.random.rand() < dynamic_accept_prob
            
            # Dynamic surge attracts additional driver capacity
            dynamic_supply = sup * (1.0 + 0.32 * (surge - 1.0))
            dynamic_fulfilled = dynamic_accepted and (np.random.rand() < min(1.0, dynamic_supply / max(1.0, dem)))
            dynamic_rev = dynamic_price if dynamic_fulfilled else 0.0

            simulation_records.append({
                "request_id": f"SIM_{i+1:04d}",
                "distance": round(dist, 2),
                "duration": round(dur, 1),
                "is_premium": int(prem),
                "demand": int(dem),
                "supply": int(sup),
                "demand_supply_ratio": round(dem / max(1.0, sup), 2),
                "surge_multiplier": surge,
                "static_price": static_price,
                "static_fulfilled": int(static_fulfilled),
                "static_revenue": static_rev,
                "dynamic_price": dynamic_price,
                "dynamic_fulfilled": int(dynamic_fulfilled),
                "dynamic_revenue": dynamic_rev
            })

        sim_df = pd.DataFrame(simulation_records)

        static_total_rev = float(sim_df["static_revenue"].sum())
        dynamic_total_rev = float(sim_df["dynamic_revenue"].sum())
        rev_lift_pct = ((dynamic_total_rev - static_total_rev) / max(1.0, static_total_rev)) * 100.0

        static_fulfilled_cnt = int(sim_df["static_fulfilled"].sum())
        dynamic_fulfilled_cnt = int(sim_df["dynamic_fulfilled"].sum())
        fulfillment_lift_pct = ((dynamic_fulfilled_cnt - static_fulfilled_cnt) / max(1, static_fulfilled_cnt)) * 100.0

        summary_metrics = {
            "total_requests": n_samples,
            "static_revenue": round(static_total_rev, 2),
            "dynamic_revenue": round(dynamic_total_rev, 2),
            "revenue_lift_pct": round(rev_lift_pct, 1),
            "static_fulfillment_rate": round((static_fulfilled_cnt / n_samples) * 100.0, 1),
            "dynamic_fulfillment_rate": round((dynamic_fulfilled_cnt / n_samples) * 100.0, 1),
            "fulfillment_lift_pct": round(fulfillment_lift_pct, 1),
            "avg_surge_multiplier": round(float(sim_df["surge_multiplier"].mean()), 2),
            "static_avg_fare": round(float(sim_df["static_price"].mean()), 2),
            "dynamic_avg_fare": round(float(sim_df["dynamic_price"].mean()), 2)
        }

        return {
            "summary": summary_metrics,
            "sim_df": sim_df
        }


if __name__ == "__main__":
    optimizer = PriceOptimizer()
    print("Testing ride surge optimization:")
    res = optimizer.optimize_ride_surge(demand=80, supply=30, distance_miles=4.0)
    print(f"Optimal Surge: {res['optimal_surge']}x | Expected Revenue: ${res['optimal_expected_revenue']}")
    print(f"Rule-based Surge: {res['rule_based_surge']}x")
    print("\nRunning quick marketplace simulation:")
    sim = optimizer.run_marketplace_simulation(n_samples=500)
    print("Simulation Summary:", sim["summary"])
