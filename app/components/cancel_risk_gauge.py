"""
Cancellation Risk Gauge & Elasticity UI Component.
Visualizes customer rejection / cancellation probability, status zones, and price sensitivity.
"""

import streamlit as st


def render_cancellation_risk_gauge(risk_prob: float, cap_threshold: float = 0.45):
    """
    Render customer drop-off risk card with animated progress bar and status indicator.
    """
    risk_pct = round(risk_prob * 100.0, 1)
    
    if risk_prob < 0.20:
        status_label = "Optimal Conversion (Low Drop-off Risk)"
        status_color = "#10b981"
        bar_color = "linear-gradient(90deg, #10b981, #34d399)"
        advice = "Demand is resilient. Surge is well-calibrated for high marketplace booking velocity."
    elif risk_prob < 0.35:
        status_label = "Moderate Friction (Healthy Elasticity)"
        status_color = "#f59e0b"
        bar_color = "linear-gradient(90deg, #10b981, #f59e0b)"
        advice = "Slight rider resistance observed. Revenue per trip remains high."
    elif risk_prob <= cap_threshold:
        status_label = "High Drop-off Risk (Nearing Policy Cap)"
        status_color = "#f97316"
        bar_color = "linear-gradient(90deg, #f59e0b, #f97316)"
        advice = "Elevated churn risk. Higher surge may depress total completed trips."
    else:
        status_label = "CRITICAL CHURN RISK (Exceeds Policy Cap)"
        status_color = "#ef4444"
        bar_color = "linear-gradient(90deg, #f97316, #ef4444)"
        advice = "Surge exceeds rider willingness-to-pay threshold. Price cap active."

    gauge_html = f"""
    <div class="glass-card" style="margin-top: 16px;">
        <div style="display: flex; justify-content: space-between; align-items: baseline; margin-bottom: 8px;">
            <span style="font-size: 0.85rem; font-weight: 700; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.05em;">Customer Drop-off Risk</span>
            <span style="font-size: 1.4rem; font-weight: 800; color: {status_color}; font-family: 'JetBrains Mono', monospace;">{risk_pct}%</span>
        </div>
        <div style="font-size: 0.9rem; font-weight: 600; color: {status_color}; margin-bottom: 10px;">
            {status_label}
        </div>
        <div class="progress-bar-container">
            <div class="progress-bar-fill" style="width: {min(100.0, risk_pct)}%; background: {bar_color};"></div>
        </div>
        <div style="display: flex; justify-content: space-between; font-size: 0.75rem; color: #64748b; margin-top: 6px;">
            <span>0% (Safe)</span>
            <span>Target Cap: {int(cap_threshold * 100)}%</span>
            <span>100% (High Churn)</span>
        </div>
        <div style="margin-top: 14px; padding: 10px; background: rgba(15, 23, 42, 0.5); border-radius: 8px; border-left: 3px solid {status_color}; font-size: 0.82rem; color: #cbd5e1;">
            💡 <strong>Optimizer Insight:</strong> {advice}
        </div>
    </div>
    """
    st.markdown(gauge_html, unsafe_allow_html=True)
