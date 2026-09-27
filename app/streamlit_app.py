"""
Dynamic Fare Engine - Simple Black & White Light Theme Application.
Page 1: Ride Charges & Fare Ticket
Page 2: Food Delivery Charges & Printed Box Bill Slip
"""

import os
import sys
from pathlib import Path

# Add project root to sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import streamlit as st
from datetime import datetime

from src.pricing.surge_engine import DynamicFareEngine
from src.db.db_utils import init_db, log_pricing_event, get_zones_df
from app.components.price_display import render_food_delivery_bill, render_ride_fare_ticket

# Page configuration
st.set_page_config(
    page_title="Dynamic Fare Engine",
    page_icon="🧾",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Load CSS
css_path = Path(__file__).resolve().parent / "assets" / "style.css"
if css_path.exists():
    with open(css_path, "r") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Initialize engine & zones
@st.cache_resource
def get_engine():
    init_db()
    return DynamicFareEngine()

engine = get_engine()
zones_df = get_zones_df()
zone_list = list(zones_df["zone_name"].values) if not zones_df.empty else ["Financial District", "Back Bay", "Fenway", "South Station"]

# App Header
st.markdown("""
<div class="app-header">
    <div style="display: flex; justify-content: space-between; align-items: center;">
        <div class="app-title">⚡ DYNAMIC FARE ENGINE</div>
        <div class="app-badge">SEMESTER PROJECT</div>
    </div>
</div>
""", unsafe_allow_html=True)

# 2 Pages Selector
page = st.radio(
    "SELECT APPLICATION:",
    ["🚗 Ride Charges", "🍕 Food Delivery Charges"],
    horizontal=True,
    label_visibility="collapsed"
)

# =====================================================================
# PAGE 1: RIDE CHARGES
# =====================================================================
if page == "🚗 Ride Charges":
    col_in, col_out = st.columns([1.1, 1.0], gap="medium")

    with col_in:
        with st.container(border=True):
            st.markdown("**1. TRIP DETAILS**")
            c1, c2 = st.columns(2)
            with c1:
                pickup = st.selectbox("Pickup Location", zone_list, index=0)
            with c2:
                dropoff = st.selectbox("Dropoff Location", zone_list, index=min(3, len(zone_list)-1))

            c_d, c_t = st.columns(2)
            with c_d:
                dist = st.number_input("Distance (miles)", min_value=0.5, max_value=50.0, value=4.5, step=0.5)
            with c_t:
                est_dur = round(dist * 3.2 + 2.0, 0)
                dur = st.number_input("Duration (minutes)", min_value=1.0, max_value=120.0, value=est_dur, step=1.0)

            st.markdown("**2. VEHICLE & DEMAND SURGE**")
            c_tier, c_surge = st.columns(2)
            with c_tier:
                tier = st.selectbox("Cab Category", ["Standard (UberX)", "Premium (Black/Lux)", "Shared (Pool)"], index=0)
                is_prem = "Premium" in tier
                is_shar = "Shared" in tier
            with c_surge:
                surge_val = st.slider("Surge Multiplier", 1.0, 3.5, 1.4, step=0.1)

    with col_out:
        quote = engine.calculate_ride_fare(
            distance_miles=dist,
            duration_min=dur,
            surge_multiplier=surge_val,
            is_premium=is_prem,
            is_shared=is_shar
        )

        trip_id = f"RD-{int(datetime.now().timestamp()) % 10000:04d}"
        render_ride_fare_ticket(
            trip_id=trip_id,
            pickup=pickup,
            dropoff=dropoff,
            ride_quote=quote
        )

        if st.button("💾 Save Trip to Database", use_container_width=True):
            log_pricing_event({
                "service_type": "ride",
                "zone_name": pickup,
                "distance_miles": dist,
                "duration_min": dur,
                "base_fare": quote["base_fare"],
                "surge_multiplier": quote["surge_multiplier"],
                "total_fare": quote["total_fare"],
                "cancellation_risk": 0.05,
                "is_premium": int(is_prem),
                "customer_accepted": 1,
                "driver_payout": quote["driver_payout"],
                "platform_fee": quote["platform_fee"]
            })
            st.success(f"Trip #{trip_id} saved successfully!")

# =====================================================================
# PAGE 2: FOOD DELIVERY CHARGES & PRINTED BOX BILL SLIP
# =====================================================================
else:
    col_in, col_out = st.columns([1.1, 1.0], gap="medium")

    with col_in:
        with st.container(border=True):
            st.markdown("**1. FOOD ORDER ITEMS**")
            
            c_rest, c_cust = st.columns(2)
            with c_rest:
                rest_name = st.text_input("Restaurant Name", value="PIZZA & BURGER CO.")
            with c_cust:
                cust_name = st.text_input("Customer Name", value="Yash S.")

            c_q1, c_q2, c_q3 = st.columns(3)
            with c_q1:
                q1 = st.number_input("Pizza (₹220)", 0, 10, 1)
            with c_q2:
                q2 = st.number_input("Burger (₹130)", 0, 10, 1)
            with c_q3:
                q3 = st.number_input("Fries/Drink (₹60)", 0, 10, 1)

            food_items = []
            if q1 > 0:
                food_items.append({"qty": q1, "name": "Cheese Loaded Pizza", "price": 220.0 * q1})
            if q2 > 0:
                food_items.append({"qty": q2, "name": "Crispy Chicken Burger", "price": 130.0 * q2})
            if q3 > 0:
                food_items.append({"qty": q3, "name": "Cold Beverage / Fries", "price": 60.0 * q3})

            food_subtotal = sum(it["price"] for it in food_items)
            if food_subtotal == 0:
                food_subtotal = 100.0
                food_items.append({"qty": 1, "name": "Meal Box", "price": 100.0})

            st.markdown(f"**Food Subtotal:** `₹{food_subtotal:.2f}`")

            st.markdown("**2. DELIVERY CONDITIONS**")
            c_dist, c_surge = st.columns(2)
            with c_dist:
                deliv_dist_km = st.number_input("Distance (km)", 0.5, 30.0, 4.5, step=0.5)
            with c_surge:
                deliv_surge = st.slider("Delivery Surge", 1.0, 2.5, 1.2, step=0.1)

            c_w, c_t = st.columns(2)
            with c_w:
                w_choice = st.selectbox("Weather", ["Clear (₹0)", "Rain (+₹10)", "Storm (+₹20)"], index=1)
                w_sev = 3 if "Storm" in w_choice else (2 if "Rain" in w_choice else 1)
            with c_t:
                t_choice = st.selectbox("Traffic", ["Normal (₹0)", "Congestion (+₹5)", "Jam (+₹10)"], index=1)
                t_sev = 3 if "Jam" in t_choice else (2 if "Congestion" in t_choice else 1)

    with col_out:
        deliv_quote = engine.calculate_delivery_fee(
            distance_km=deliv_dist_km,
            order_value=food_subtotal,
            surge_multiplier=deliv_surge,
            weather_severity=w_sev,
            traffic_severity=t_sev
        )

        order_id = f"QB-{int(datetime.now().timestamp()) % 10000:04d}"
        render_food_delivery_bill(
            order_id=order_id,
            food_items=food_items,
            food_subtotal=food_subtotal,
            delivery_quote=deliv_quote,
            customer_name=cust_name,
            restaurant_name=rest_name
        )

        if st.button("💾 Save Order to Database", use_container_width=True):
            log_pricing_event({
                "service_type": "delivery",
                "zone_name": rest_name,
                "distance_miles": deliv_dist_km * 0.621371,
                "duration_min": deliv_quote["predicted_delay_min"],
                "base_fare": deliv_quote["base_fee"],
                "surge_multiplier": deliv_quote["surge_multiplier"],
                "total_fare": deliv_quote["total_delivery_fee"],
                "cancellation_risk": 0.05,
                "customer_accepted": 1,
                "driver_payout": deliv_quote["driver_payout"],
                "platform_fee": deliv_quote["platform_fee"]
            })
            st.success(f"Order #{order_id} saved successfully!")
