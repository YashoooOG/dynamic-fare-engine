"""
Ingestion & Simulation module for Driver Supply and Fleet Availability.
Generates synthetic driver/rider availability distributions across urban zones and hours,
saving to data/processed/simulated_supply.parquet.
"""

import sys
import logging
from pathlib import Path
import pandas as pd
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_ZONES = [
    "Back Bay", "Beacon Hill", "Boston University", "Fenway",
    "Financial District", "Haymarket Square", "North End",
    "North Station", "Northeastern University", "South Station",
    "Theatre District", "West End"
]


def generate_simulated_supply(
    zones: list = None,
    days: int = 14,
    seed: int = 42,
    output_path: str = "data/processed/simulated_supply.parquet"
) -> pd.DataFrame:
    """
    Generate synthetic supply-demand marketplace states per zone and hourly timestep.
    
    Attributes:
    - timestamp, zone, hour, day_of_week
    - active_drivers: Available supply
    - incoming_ride_requests: Observed demand
    - demand_supply_ratio: Demand / Supply pressure
    - avg_driver_wait_time_min: Estimated pickup wait time
    - fleet_utilization_rate: Proportion of drivers engaged in trips
    """
    np.random.seed(seed)
    zones = zones or DEFAULT_ZONES
    
    logger.info(f"Generating synthetic fleet supply for {len(zones)} zones over {days} days...")
    
    records = []
    start_date = pd.Timestamp("2026-08-01 00:00:00")
    total_hours = days * 24
    
    for h_idx in range(total_hours):
        current_dt = start_date + pd.Timedelta(hours=h_idx)
        hour = current_dt.hour
        day_of_week = current_dt.dayofweek
        is_weekend = int(day_of_week in [5, 6])
        
        # Diurnal base demand factor (peaks at 8 AM and 6 PM)
        time_factor = 1.0 + 0.6 * np.exp(-((hour - 8.5) ** 2) / 8.0) + 0.8 * np.exp(-((hour - 18.0) ** 2) / 10.0)
        if is_weekend and (hour in [22, 23, 0, 1, 2]):
            time_factor += 0.7  # Nightlife surge
            
        for zone in zones:
            # Zone popularity factor
            zone_bias = 1.2 if zone in ["Financial District", "Back Bay", "South Station", "Fenway"] else 0.9
            
            # Simulated demand and supply
            expected_demand = int(np.clip(np.random.poisson(lam=45 * time_factor * zone_bias), 5, 200))
            expected_supply = int(np.clip(np.random.normal(loc=35 * (0.8 + 0.4 * np.sin(hour / 24.0 * np.pi)), scale=6), 4, 120))
            
            ratio = expected_demand / max(1, expected_supply)
            utilization = float(np.clip(ratio * 0.75, 0.20, 0.98))
            wait_time = float(np.clip(2.5 + 3.0 * (ratio - 0.8), 1.5, 18.0))
            
            records.append({
                "timestamp": current_dt,
                "zone": zone,
                "hour": hour,
                "day_of_week": day_of_week,
                "is_weekend": is_weekend,
                "active_drivers": expected_supply,
                "incoming_ride_requests": expected_demand,
                "demand_supply_ratio": round(ratio, 3),
                "avg_driver_wait_time_min": round(wait_time, 2),
                "fleet_utilization_rate": round(utilization, 3)
            })
            
    df = pd.DataFrame(records)
    
    # Save output
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False, engine="pyarrow")
    
    logger.info(f"Successfully generated {len(df):,} supply records and saved to {output_path}")
    return df


if __name__ == "__main__":
    out_file = sys.argv[1] if len(sys.argv) > 1 else "data/processed/simulated_supply.parquet"
    generate_simulated_supply(output_path=out_file)
