"""
Database Access Layer and Utility Functions for Dynamic Fare Engine.
Provides thread-safe connections, event logging, snapshot persistence, and analytics queries.
"""

import os
import sqlite3
import logging
from pathlib import Path
from datetime import datetime, timezone
import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

DB_PATH = os.getenv("DB_PATH", "data/db.sqlite")
DB_URL = os.getenv("DB_URL", f"sqlite:///{DB_PATH}")

engine = create_engine(DB_URL, connect_args={"check_same_thread": False})

BOSTON_ZONES = [
    {"zone_name": "Back Bay", "latitude": 42.3503, "longitude": -71.0810, "base_demand_multiplier": 1.20},
    {"zone_name": "Beacon Hill", "latitude": 42.3588, "longitude": -71.0707, "base_demand_multiplier": 1.05},
    {"zone_name": "Boston University", "latitude": 42.3505, "longitude": -71.1054, "base_demand_multiplier": 1.15},
    {"zone_name": "Fenway", "latitude": 42.3429, "longitude": -71.0988, "base_demand_multiplier": 1.25},
    {"zone_name": "Financial District", "latitude": 42.3559, "longitude": -71.0550, "base_demand_multiplier": 1.35},
    {"zone_name": "Haymarket Square", "latitude": 42.3628, "longitude": -71.0583, "base_demand_multiplier": 1.00},
    {"zone_name": "North End", "latitude": 42.3647, "longitude": -71.0542, "base_demand_multiplier": 1.10},
    {"zone_name": "North Station", "latitude": 42.3664, "longitude": -71.0620, "base_demand_multiplier": 1.20},
    {"zone_name": "Northeastern University", "latitude": 42.3398, "longitude": -71.0892, "base_demand_multiplier": 1.10},
    {"zone_name": "South Station", "latitude": 42.3523, "longitude": -71.0552, "base_demand_multiplier": 1.30},
    {"zone_name": "Theatre District", "latitude": 42.3519, "longitude": -71.0643, "base_demand_multiplier": 1.15},
    {"zone_name": "West End", "latitude": 42.3644, "longitude": -71.0661, "base_demand_multiplier": 0.95}
]


def init_db(schema_path: str = "src/db/schema.sql") -> None:
    """Initialize database tables and seed default urban zones if empty."""
    db_file = Path(DB_PATH)
    db_file.parent.mkdir(parents=True, exist_ok=True)
    
    schema_file = Path(schema_path)
    if not schema_file.exists():
        logger.error(f"Schema file not found at: {schema_path}")
        return

    with open(schema_file, "r") as f:
        schema_sql = f.read()

    with engine.begin() as conn:
        for statement in schema_sql.split(";"):
            stmt_clean = statement.strip()
            if stmt_clean:
                conn.execute(text(stmt_clean))
                
        # Seed default zones if empty
        result = conn.execute(text("SELECT COUNT(*) FROM zones")).scalar()
        if result == 0:
            logger.info("Seeding initial urban zones catalog...")
            for z in BOSTON_ZONES:
                conn.execute(
                    text("""
                        INSERT OR IGNORE INTO zones (zone_name, latitude, longitude, base_demand_multiplier)
                        VALUES (:zone_name, :latitude, :longitude, :base_demand_multiplier)
                    """),
                    z
                )
    logger.info("Database initialized and verified successfully.")


def log_pricing_event(event_dict: dict) -> str:
    """Persist an individual ride or delivery pricing quote to SQLite."""
    request_id = event_dict.get("request_id") or f"REQ_{int(datetime.now(timezone.utc).timestamp() * 1000)}"
    service_type = event_dict.get("service_type", "ride")
    timestamp = event_dict.get("timestamp", datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"))
    zone_name = event_dict.get("zone_name", "Financial District")
    distance_miles = float(event_dict.get("distance_miles", 0.0))
    duration_min = float(event_dict.get("duration_min", 0.0))
    base_fare = float(event_dict.get("base_fare", 0.0))
    surge_multiplier = float(event_dict.get("surge_multiplier", 1.0))
    total_fare = float(event_dict.get("total_fare", 0.0))
    is_premium = int(event_dict.get("is_premium", 0))
    weather_condition = str(event_dict.get("weather_condition", "Clear"))
    traffic_condition = str(event_dict.get("traffic_condition", "Normal"))
    customer_accepted = int(event_dict.get("customer_accepted", 1))
    driver_payout = float(event_dict.get("driver_payout", total_fare * 0.78))
    platform_fee = float(event_dict.get("platform_fee", total_fare * 0.22))

    query = text("""
        INSERT INTO pricing_logs (
            request_id, service_type, timestamp, zone_name, distance_miles,
            duration_min, base_fare, surge_multiplier, total_fare,
            is_premium, weather_condition,
            traffic_condition, customer_accepted, driver_payout, platform_fee
        ) VALUES (
            :request_id, :service_type, :timestamp, :zone_name, :distance_miles,
            :duration_min, :base_fare, :surge_multiplier, :total_fare,
            :is_premium, :weather_condition,
            :traffic_condition, :customer_accepted, :driver_payout, :platform_fee
        )
    """)

    with engine.begin() as conn:
        conn.execute(query, {
            "request_id": request_id,
            "service_type": service_type,
            "timestamp": timestamp,
            "zone_name": zone_name,
            "distance_miles": distance_miles,
            "duration_min": duration_min,
            "base_fare": base_fare,
            "surge_multiplier": surge_multiplier,
            "total_fare": total_fare,
            "is_premium": is_premium,
            "weather_condition": weather_condition,
            "traffic_condition": traffic_condition,
            "customer_accepted": customer_accepted,
            "driver_payout": driver_payout,
            "platform_fee": platform_fee
        })

    return request_id


def get_recent_pricing_logs(limit: int = 50) -> pd.DataFrame:
    """Retrieve recent pricing quotes."""
    query = f"SELECT * FROM pricing_logs ORDER BY timestamp DESC LIMIT {limit}"
    try:
        return pd.read_sql(query, con=engine)
    except Exception as e:
        logger.warning(f"Failed to query pricing logs: {e}")
        return pd.DataFrame()


def get_zones_df() -> pd.DataFrame:
    """Fetch all registered urban zones with geospatial coordinates."""
    try:
        df = pd.read_sql("SELECT * FROM zones ORDER BY zone_name", con=engine)
        if df.empty:
            init_db()
            df = pd.read_sql("SELECT * FROM zones ORDER BY zone_name", con=engine)
        return df
    except Exception:
        init_db()
        return pd.DataFrame(BOSTON_ZONES)


def get_marketplace_kpis() -> dict:
    """Compute aggregate marketplace operational metrics from logs."""
    try:
        df = pd.read_sql("SELECT * FROM pricing_logs", con=engine)
        if df.empty:
            return {
                "total_quotes": 0,
                "accepted_rides": 0,
                "acceptance_rate_pct": 100.0,
                "gross_revenue": 0.0,
                "avg_surge": 1.0,
                "avg_fare": 0.0
            }
        
        total = len(df)
        accepted = int(df["customer_accepted"].sum())
        revenue = float(df[df["customer_accepted"] == 1]["total_fare"].sum())
        
        return {
            "total_quotes": total,
            "accepted_rides": accepted,
            "acceptance_rate_pct": round((accepted / max(1, total)) * 100, 1),
            "gross_revenue": round(revenue, 2),
            "avg_surge": round(float(df["surge_multiplier"].mean()), 2),
            "avg_fare": round(float(df["total_fare"].mean()), 2)
        }
    except Exception as e:
        logger.warning(f"Error computing KPIs: {e}")
        return {
            "total_quotes": 0,
            "accepted_rides": 0,
            "acceptance_rate_pct": 0.0,
            "gross_revenue": 0.0,
            "avg_surge": 1.0,
            "avg_fare": 0.0
        }


if __name__ == "__main__":
    init_db()
    print("Database initial verification successful.")
