-- =====================================================================
-- Database Schema for Dynamic Fare Engine
-- SQLite Compatible
-- =====================================================================

-- 1. Urban Zones catalog
CREATE TABLE IF NOT EXISTS zones (
    zone_id INTEGER PRIMARY KEY AUTOINCREMENT,
    zone_name VARCHAR(64) UNIQUE NOT NULL,
    latitude REAL NOT NULL,
    longitude REAL NOT NULL,
    base_demand_multiplier REAL DEFAULT 1.0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 2. Pricing Request & Quote Event Logs
CREATE TABLE IF NOT EXISTS pricing_logs (
    request_id VARCHAR(64) PRIMARY KEY,
    service_type VARCHAR(32) NOT NULL, -- 'ride' or 'delivery'
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    zone_name VARCHAR(64),
    distance_miles REAL,
    duration_min REAL,
    base_fare REAL NOT NULL,
    surge_multiplier REAL NOT NULL,
    total_fare REAL NOT NULL,
    is_premium INTEGER DEFAULT 0,
    weather_condition VARCHAR(32),
    traffic_condition VARCHAR(32),
    customer_accepted INTEGER DEFAULT 1,
    driver_payout REAL,
    platform_fee REAL
);

-- 3. Marketplace Time-series Supply & Demand Snapshots
CREATE TABLE IF NOT EXISTS marketplace_snapshots (
    snapshot_id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TIMESTAMP NOT NULL,
    zone_name VARCHAR(64) NOT NULL,
    hour INTEGER NOT NULL,
    day_of_week INTEGER NOT NULL,
    active_drivers INTEGER NOT NULL,
    incoming_requests INTEGER NOT NULL,
    demand_supply_ratio REAL NOT NULL,
    avg_wait_time_min REAL NOT NULL,
    calculated_surge REAL NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 4. Indices for fast queries and analytics
CREATE INDEX IF NOT EXISTS idx_pricing_logs_time ON pricing_logs(timestamp);
CREATE INDEX IF NOT EXISTS idx_pricing_logs_zone ON pricing_logs(zone_name);
CREATE INDEX IF NOT EXISTS idx_snapshots_zone_time ON marketplace_snapshots(zone_name, timestamp);
