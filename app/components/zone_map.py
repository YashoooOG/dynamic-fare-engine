"""
Urban Zone Map & Real-time Geospatial Surge Visualizer Component.
Displays interactive geographic points, zone surge multipliers, and fleet utilization.
"""

import streamlit as st
import pandas as pd
import numpy as np


def render_zone_map_and_cards(zones_df: pd.DataFrame, current_hour: int = 14):
    """
    Render geospatial map of urban zones with real-time dynamic surge indicators.
    """
    if zones_df.empty:
        st.warning("No zone geospatial coordinates found.")
        return

    # Derive real-time simulated status per zone for the given hour
    time_factor = 1.0 + 0.6 * np.exp(-((current_hour - 8.5) ** 2) / 8.0) + 0.8 * np.exp(-((current_hour - 18.0) ** 2) / 10.0)
    
    records = []
    for _, row in zones_df.iterrows():
        base_mult = float(row.get("base_demand_multiplier", 1.0))
        dem = int(np.clip(45 * time_factor * base_mult, 10, 180))
        sup = int(np.clip(35 * (0.8 + 0.4 * np.sin(current_hour / 24.0 * np.pi)), 10, 100))
        ratio = round(dem / max(1, sup), 2)
        surge = 1.0 if ratio <= 1.0 else round(min(3.5, 1.0 + 0.85 * (ratio - 1.0)), 2)
        wait_min = round(float(np.clip(2.5 + 3.0 * (ratio - 0.8), 1.5, 15.0)), 1)
        
        records.append({
            "zone_name": row["zone_name"],
            "lat": float(row["latitude"]),
            "lon": float(row["longitude"]),
            "demand": dem,
            "supply": sup,
            "ratio": ratio,
            "surge": surge,
            "wait_time": wait_min
        })

    status_df = pd.DataFrame(records)

    # 1. Streamlit map
    st.map(status_df[["lat", "lon"]], zoom=12, use_container_width=True)

    # 2. Zone Status Matrix Grid
    st.markdown("#### 🏙️ Real-Time Zone Dispatch Status")
    
    cols = st.columns(3)
    for idx, row in status_df.iterrows():
        col = cols[idx % 3]
        s_color = "#34d399" if row["surge"] <= 1.05 else ("#fbbf24" if row["surge"] < 1.8 else "#fb7185")
        
        with col:
            st.markdown(f"""
            <div class="glass-card" style="padding: 16px; margin-bottom: 12px;">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <span style="font-weight: 700; font-size: 0.95rem; color: #f8fafc;">{row['zone_name']}</span>
                    <span style="color: {s_color}; font-weight: 800; font-family: 'JetBrains Mono', monospace;">{row['surge']}x</span>
                </div>
                <div style="display: flex; justify-content: space-between; font-size: 0.8rem; color: #94a3b8; margin-top: 8px;">
                    <span>Demand: <strong style="color: #cbd5e1;">{row['demand']}</strong></span>
                    <span>Drivers: <strong style="color: #cbd5e1;">{row['supply']}</strong></span>
                    <span>Wait: <strong style="color: #cbd5e1;">{row['wait_time']}m</strong></span>
                </div>
            </div>
            """, unsafe_allow_html=True)
