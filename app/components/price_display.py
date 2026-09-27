"""
Thermal Bill Slip & Printed Receipt UI Component.
Renders authentic thermal printed receipts using clean native monospace formatting.
"""

import streamlit as st
from datetime import datetime


def render_food_delivery_bill(
    order_id: str,
    food_items: list,
    food_subtotal: float,
    delivery_quote: dict,
    customer_name: str = "Customer",
    customer_address: str = "Flat 402, Green Park",
    restaurant_name: str = "PIZZA & BURGER CO."
):
    """
    Render realistic printed thermal bill slip taped onto delivery food boxes.
    """
    now_str = datetime.now().strftime("%d-%b-%Y %I:%M %p")
    deliv_fee = delivery_quote.get("total_delivery_fee", 0.0)
    grand_total = food_subtotal + deliv_fee
    surge = delivery_quote.get("surge_multiplier", 1.0)
    weather_fee = delivery_quote.get("weather_surcharge", 0.0)
    traffic_fee = delivery_quote.get("traffic_surcharge", 0.0)
    small_basket = delivery_quote.get("small_basket_surcharge", 0.0)
    dist_fee = delivery_quote.get("distance_fee", 0.0)
    base_fee = delivery_quote.get("base_fee", 30.0)
    dist_km = delivery_quote.get("distance_km", 0.0)
    est_delay = delivery_quote.get("predicted_delay_min", 20.0)

    # Build items lines (fixed 40 characters width)
    item_lines = []
    for item in food_items:
        name_str = f"{item['qty']}x {item['name']}"[:26]
        price_str = f"INR {item['price']:.2f}"
        space_count = max(1, 40 - len(name_str) - len(price_str))
        item_lines.append(f"{name_str}{' ' * space_count}{price_str}")
    items_block = "\n".join(item_lines)

    # Build optional surcharges lines
    surcharge_lines = []
    if weather_fee > 0:
        w_val = f"+INR {weather_fee:.2f}"
        surcharge_lines.append(f"Weather Surcharge (Rain){' ' * max(1, 40 - 24 - len(w_val))}{w_val}")
    if traffic_fee > 0:
        t_val = f"+INR {traffic_fee:.2f}"
        surcharge_lines.append(f"Traffic Jam Surcharge{' ' * max(1, 40 - 21 - len(t_val))}{t_val}")
    if small_basket > 0:
        b_val = f"+INR {small_basket:.2f}"
        surcharge_lines.append(f"Small Basket Fee (<INR 250){' ' * max(1, 40 - 26 - len(b_val))}{b_val}")
    if surge > 1.0:
        s_val = f"{surge:.1f}x"
        surcharge_lines.append(f"Peak Demand Surge{' ' * max(1, 40 - 18 - len(s_val))}{s_val}")

    surcharges_block = "\n".join(surcharge_lines)
    if surcharges_block:
        surcharges_block += "\n"

    base_str = f"INR {base_fee:.2f}"
    dist_str = f"INR {dist_fee:.2f}"
    deliv_tot_str = f"INR {deliv_fee:.2f}"
    grand_tot_str = f"INR {grand_total:.2f}"
    food_sub_str = f"INR {food_subtotal:.2f}"

    base_line = f"Base Delivery Fee{' ' * max(1, 40 - 17 - len(base_str))}{base_str}"
    dist_label = f"Distance Fee ({dist_km:.1f} km)"
    dist_line = f"{dist_label}{' ' * max(1, 40 - len(dist_label) - len(dist_str))}{dist_str}"
    deliv_tot_line = f"TOTAL DELIVERY FEE:{' ' * max(1, 40 - 19 - len(deliv_tot_str))}{deliv_tot_str}"
    grand_tot_line = f"GRAND TOTAL:{' ' * max(1, 40 - 12 - len(grand_tot_str))}{grand_tot_str}"
    food_sub_line = f"FOOD SUBTOTAL:{' ' * max(1, 40 - 14 - len(food_sub_str))}{food_sub_str}"

    receipt_text = f"""========================================
          PACKAGE DELIVERY SLIP
========================================
STORE:    {restaurant_name}
ORDER:    #{order_id}
DATE:     {now_str}
DELIVER:  {customer_name}
ADDRESS:  {customer_address}
----------------------------------------
ITEM                               PRICE
----------------------------------------
{items_block}
----------------------------------------
{food_sub_line}
----------------------------------------
DELIVERY CHARGES BREAKDOWN:
{base_line}
{dist_line}
{surcharges_block}----------------------------------------
{deliv_tot_line}
========================================
{grand_tot_line}
========================================
         ||| | |||| || | ||| ||
        EST. DELIVERY: ~{est_delay:.0f} MINS
      THANK YOU FOR YOUR ORDER!
========================================"""

    st.code(receipt_text, language="text")


def render_ride_fare_ticket(
    trip_id: str,
    pickup: str,
    dropoff: str,
    ride_quote: dict
):
    """
    Render clean printed ride-hailing ticket.
    """
    now_str = datetime.now().strftime("%d-%b-%Y %I:%M %p")
    total_fare = ride_quote.get("total_fare", 0.0)
    surge = ride_quote.get("surge_multiplier", 1.0)
    dist = ride_quote.get("distance_miles", 0.0)
    dur = ride_quote.get("duration_min", 0.0)
    base_fare = ride_quote.get("base_fare", 3.50)
    dist_cost = ride_quote.get("distance_cost", 0.0)
    time_cost = ride_quote.get("time_cost", 0.0)
    driver_payout = ride_quote.get("driver_payout", 0.0)
    platform_fee = ride_quote.get("platform_fee", 0.0)
    tier = ride_quote.get("tier_label", "Standard")

    surge_str = ""
    if surge > 1.0:
        s_val = f"{surge:.1f}x"
        surge_str = f"Surge Multiplier:{' ' * max(1, 40 - 17 - len(s_val))}{s_val}\n"

    base_val = f"${base_fare:.2f}"
    dist_val = f"${dist_cost:.2f}"
    time_val = f"${time_cost:.2f}"
    tot_val = f"${total_fare:.2f}"
    driver_val = f"${driver_payout:.2f}"
    plat_val = f"${platform_fee:.2f}"

    base_line = f"Base Fare:{' ' * max(1, 40 - 10 - len(base_val))}{base_val}"
    dist_lbl = f"Distance ({dist:.1f} mi @ $1.85):"
    dist_line = f"{dist_lbl}{' ' * max(1, 40 - len(dist_lbl) - len(dist_val))}{dist_val}"
    time_lbl = f"Time ({dur:.0f} min @ $0.35):"
    time_line = f"{time_lbl}{' ' * max(1, 40 - len(time_lbl) - len(time_val))}{time_val}"
    total_line = f"TOTAL FARE:{' ' * max(1, 40 - 11 - len(tot_val))}{tot_val}"
    driver_line = f"Driver Payout (78%):{' ' * max(1, 40 - 20 - len(driver_val))}{driver_val}"
    plat_line = f"Platform Fee (22%):{' ' * max(1, 40 - 19 - len(plat_val))}{plat_val}"

    ticket_text = f"""========================================
          RIDE FARE TICKET
========================================
DISPATCH: DYNAMIC RIDE ENGINE
TICKET:   #{trip_id}
DATE:     {now_str}
FROM:     {pickup}
TO:       {dropoff}
TIER:     {tier}
----------------------------------------
FARE CALCULATION BREAKDOWN:
{base_line}
{dist_line}
{time_line}
{surge_str}----------------------------------------
{total_line}
========================================
{driver_line}
{plat_line}
----------------------------------------
          || ||| || | |||| |||
           HAVE A SAFE TRIP!
========================================"""

    st.code(ticket_text, language="text")
