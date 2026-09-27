"""
Marketplace Demand, Supply, and Revenue Curves Chart Component.
Renders responsive time-series curves, price elasticity plots, and Monte Carlo comparison visualizers.
"""

import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import matplotlib.pyplot as plt
import seaborn as sns


def render_hourly_demand_supply_chart(current_hour: int = 14):
    """
    Render 24-hour diurnal demand vs. supply curve with current time marker.
    """
    hours = np.arange(0, 24)
    time_factor = 1.0 + 0.6 * np.exp(-((hours - 8.5) ** 2) / 8.0) + 0.8 * np.exp(-((hours - 18.0) ** 2) / 10.0)
    
    demand = (45 * time_factor * 1.15).astype(int)
    supply = (35 * (0.8 + 0.4 * np.sin(hours / 24.0 * np.pi))).astype(int)
    
    surges = []
    for d, s in zip(demand, supply):
        ratio = d / max(1, s)
        surge = 1.0 if ratio <= 1.0 else 1.0 + 0.85 * (ratio - 1.0)
        surges.append(round(min(3.5, surge), 2))

    df = pd.DataFrame({
        "Hour of Day": hours,
        "Incoming Ride Demand": demand,
        "Active Driver Supply": supply,
        "Surge Multiplier": surges
    })

    melted = df.melt(
        id_vars=["Hour of Day"],
        value_vars=["Incoming Ride Demand", "Active Driver Supply"],
        var_name="Metric",
        value_name="Count"
    )

    chart = alt.Chart(melted).mark_line(point=True, strokeWidth=3).encode(
        x=alt.X("Hour of Day:Q", title="Hour of Day (00:00 - 23:00)", axis=alt.Axis(tickCount=12)),
        y=alt.Y("Count:Q", title="Fleet / Ride Volume"),
        color=alt.Color(
            "Metric:N",
            scale=alt.Scale(
                domain=["Incoming Ride Demand", "Active Driver Supply"],
                range=["#6366f1", "#10b981"]
            ),
            legend=alt.Legend(orient="top", title=None)
        ),
        tooltip=["Hour of Day", "Metric", "Count"]
    ).properties(height=300).configure_view(strokeWidth=0).configure_axis(
        gridColor="#1e293b",
        labelColor="#94a3b8",
        titleColor="#cbd5e1"
    )

    st.altair_chart(chart, use_container_width=True)


def render_elasticity_curve(optimizer_curve_df: pd.DataFrame, selected_surge: float = 1.5):
    """
    Render revenue optimization trade-off: Expected Revenue vs. Customer Acceptance Probability.
    """
    if optimizer_curve_df.empty:
        return

    base = alt.Chart(optimizer_curve_df).encode(x=alt.X("surge_multiplier:Q", title="Surge Multiplier"))

    line_rev = base.mark_line(color="#6366f1", strokeWidth=3.5).encode(
        y=alt.Y("expected_revenue:Q", title="Expected Gross Revenue ($)", axis=alt.Axis(titleColor="#6366f1")),
        tooltip=["surge_multiplier", "expected_revenue", "acceptance_probability", "fare"]
    )

    line_accept = base.mark_line(color="#10b981", strokeDash=[4, 4], strokeWidth=2).encode(
        y=alt.Y("acceptance_probability:Q", title="Acceptance Probability", axis=alt.Axis(titleColor="#10b981", format="%")),
        tooltip=["surge_multiplier", "acceptance_probability"]
    )

    rule = alt.Chart(pd.DataFrame({"surge": [selected_surge]})).mark_rule(
        color="#38bdf8",
        strokeWidth=2,
        strokeDash=[3, 3]
    ).encode(x="surge:Q")

    combined = alt.layer(line_rev, line_accept, rule).resolve_scale(
        y="independent"
    ).properties(height=280).configure_view(strokeWidth=0)

    st.altair_chart(combined, use_container_width=True)


def render_simulation_benchmark_plots(sim_df: pd.DataFrame):
    """
    Plot static vs dynamic revenue and fulfillment distributions.
    """
    static_rev = sim_df["static_revenue"].sum()
    dynamic_rev = sim_df["dynamic_revenue"].sum()
    
    static_fulfill = sim_df["static_fulfilled"].mean() * 100.0
    dynamic_fulfill = sim_df["dynamic_fulfilled"].mean() * 100.0

    col1, col2 = st.columns(2)

    with col1:
        rev_df = pd.DataFrame({
            "Pricing Strategy": ["Static (1.0x)", "Dynamic Surge"],
            "Total Revenue ($)": [static_rev, dynamic_rev]
        })
        chart_rev = alt.Chart(rev_df).mark_bar(cornerRadiusTopLeft=8, cornerRadiusTopRight=8).encode(
            x=alt.X("Pricing Strategy:N", title=None, axis=alt.Axis(labelAngle=0)),
            y=alt.Y("Total Revenue ($):Q", title="Total Gross Revenue ($)"),
            color=alt.Color(
                "Pricing Strategy:N",
                scale=alt.Scale(domain=["Static (1.0x)", "Dynamic Surge"], range=["#64748b", "#6366f1"]),
                legend=None
            ),
            tooltip=["Pricing Strategy", "Total Revenue ($)"]
        ).properties(height=260)
        st.altair_chart(chart_rev, use_container_width=True)

    with col2:
        ful_df = pd.DataFrame({
            "Pricing Strategy": ["Static (1.0x)", "Dynamic Surge"],
            "Fulfillment Rate (%)": [static_fulfill, dynamic_fulfill]
        })
        chart_ful = alt.Chart(ful_df).mark_bar(cornerRadiusTopLeft=8, cornerRadiusTopRight=8).encode(
            x=alt.X("Pricing Strategy:N", title=None, axis=alt.Axis(labelAngle=0)),
            y=alt.Y("Fulfillment Rate (%):Q", title="Marketplace Fulfillment Rate (%)"),
            color=alt.Color(
                "Pricing Strategy:N",
                scale=alt.Scale(domain=["Static (1.0x)", "Dynamic Surge"], range=["#64748b", "#10b981"]),
                legend=None
            ),
            tooltip=["Pricing Strategy", "Fulfillment Rate (%)"]
        ).properties(height=260)
        st.altair_chart(chart_ful, use_container_width=True)
