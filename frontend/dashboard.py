import os
from datetime import datetime

import pandas as pd
import requests
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# =========================================================
# 1. APPLICATION & PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="MachineGuard Pro | Predictive Maintenance",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

API_BASE_URL = os.getenv(
    "PREDICTIVE_MAINTENANCE_API",
    "http://127.0.0.1:8000",
).rstrip("/")

# =========================================================
# 2. MODERN PREMIUM DESIGN SYSTEM & CSS
# =========================================================
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');

    :root {
        --bg-main: #0B0F19;
        --card-bg: #111827;
        --card-border: #1F2937;
        --card-hover: #1E293B;
        --primary: #38BDF8;
        --primary-glow: rgba(56, 189, 248, 0.25);
        --accent-emerald: #10B981;
        --accent-amber: #F59E0B;
        --accent-rose: #F43F5E;
        --text-primary: #F9FAFB;
        --text-secondary: #94A3B8;
        --text-muted: #64748B;
        --gradient-hero: linear-gradient(135deg, #0F172A 0%, #1E293B 50%, #0F2847 100%);
    }

    /* Overall App background */
    .stApp {
        background-color: var(--bg-main) !important;
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif !important;
        color: var(--text-primary) !important;
    }

    /* Typography */
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Plus Jakarta Sans', sans-serif !important;
        font-weight: 700 !important;
        letter-spacing: -0.02em !important;
        color: #F8FAFC !important;
    }

    p, span, label, div {
        color: var(--text-secondary);
    }

    /* Top Hero Header */
    .hero-container {
        background: var(--gradient-hero);
        border: 1px solid rgba(56, 189, 248, 0.15);
        border-radius: 18px;
        padding: 32px 36px;
        margin-bottom: 28px;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.45), 0 0 40px rgba(56, 189, 248, 0.08);
        position: relative;
        overflow: hidden;
    }

    .hero-container::before {
        content: "";
        position: absolute;
        top: -60px;
        right: -60px;
        width: 200px;
        height: 200px;
        background: radial-gradient(circle, rgba(56, 189, 248, 0.2) 0%, transparent 70%);
        pointer-events: none;
    }

    .hero-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: rgba(56, 189, 248, 0.12);
        color: #38BDF8 !important;
        border: 1px solid rgba(56, 189, 248, 0.3);
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 11px;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-bottom: 12px;
    }

    .hero-title {
        font-size: 32px !important;
        color: #FFFFFF !important;
        margin: 0 0 8px 0 !important;
        line-height: 1.2;
    }

    .hero-desc {
        color: #94A3B8 !important;
        font-size: 15px !important;
        margin: 0 !important;
        max-width: 650px;
        line-height: 1.6;
    }

    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background-color: #0A0E1A !important;
        border-right: 1px solid #1E293B !important;
    }

    [data-testid="stSidebar"] h1, 
    [data-testid="stSidebar"] h2, 
    [data-testid="stSidebar"] h3 {
        color: #F8FAFC !important;
    }

    /* Metric cards styling */
    [data-testid="stMetric"] {
        background-color: #111827 !important;
        border: 1px solid #1F2937 !important;
        border-radius: 14px !important;
        padding: 18px 22px !important;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.2) !important;
        transition: transform 0.2s ease, border-color 0.2s ease;
    }

    [data-testid="stMetric"]:hover {
        border-color: rgba(56, 189, 248, 0.4) !important;
        transform: translateY(-2px);
    }

    [data-testid="stMetricLabel"] p {
        color: #94A3B8 !important;
        font-size: 12px !important;
        font-weight: 600 !important;
        text-transform: uppercase !important;
        letter-spacing: 0.05em !important;
    }

    [data-testid="stMetricValue"] div {
        color: #38BDF8 !important;
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 26px !important;
        font-weight: 700 !important;
    }

    /* Custom Risk Result Cards */
    .risk-badge-card {
        border-radius: 16px;
        padding: 24px;
        margin: 16px 0 24px 0;
        position: relative;
        backdrop-filter: blur(10px);
    }

    .risk-critical {
        background: linear-gradient(135deg, rgba(244, 63, 94, 0.12) 0%, rgba(17, 24, 39, 0.95) 100%);
        border: 1px solid rgba(244, 63, 94, 0.4);
        border-left: 6px solid #F43F5E;
    }

    .risk-warning {
        background: linear-gradient(135deg, rgba(245, 158, 11, 0.12) 0%, rgba(17, 24, 39, 0.95) 100%);
        border: 1px solid rgba(245, 158, 11, 0.4);
        border-left: 6px solid #F59E0B;
    }

    .risk-nominal {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.12) 0%, rgba(17, 24, 39, 0.95) 100%);
        border: 1px solid rgba(16, 185, 129, 0.4);
        border-left: 6px solid #10B981;
    }

    .risk-status-tag {
        font-size: 11px;
        font-weight: 800;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        margin-bottom: 6px;
    }

    .risk-title-val {
        font-size: 26px;
        font-weight: 800;
        color: #FFFFFF;
        margin: 0 0 10px 0;
        display: flex;
        align-items: center;
        gap: 10px;
    }

    .risk-detail-text {
        font-size: 14px;
        color: #CBD5E1;
        line-height: 1.5;
    }

    /* Modern Buttons */
    .stButton > button, .stFormSubmitButton > button {
        background: linear-gradient(135deg, #0284C7 0%, #0369A1 100%) !important;
        color: #FFFFFF !important;
        border: 1px solid rgba(56, 189, 248, 0.3) !important;
        border-radius: 10px !important;
        padding: 10px 24px !important;
        font-weight: 600 !important;
        font-size: 14px !important;
        box-shadow: 0 4px 14px rgba(2, 132, 199, 0.35) !important;
        transition: all 0.2s ease !important;
    }

    .stButton > button:hover, .stFormSubmitButton > button:hover {
        background: linear-gradient(135deg, #0EA5E9 0%, #0284C7 100%) !important;
        transform: translateY(-1px) !important;
        box-shadow: 0 6px 20px rgba(2, 132, 199, 0.5) !important;
    }

    /* Containers */
    [data-testid="stForm"], [data-testid="stVerticalBlockBorderWrapper"] {
        background-color: #111827 !important;
        border: 1px solid #1F2937 !important;
        border-radius: 14px !important;
    }

    /* Inputs */
    .stTextInput input, .stNumberInput input, .stSelectbox select {
        background-color: #1F2937 !important;
        color: #F8FAFC !important;
        border: 1px solid #374151 !important;
        border-radius: 8px !important;
    }

    /* Radio Navigation item styling */
    [data-testid="stSidebar"] [data-testid="stRadio"] label {
        color: #CBD5E1 !important;
        font-weight: 500 !important;
    }

    /* Footer */
    .pro-footer {
        text-align: center;
        padding: 32px 0 16px 0;
        font-size: 13px;
        color: #475569;
        border-top: 1px solid #1E293B;
        margin-top: 40px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# =========================================================
# 3. API SERVICE CONNECTIONS & CACHING
# =========================================================
@st.cache_data(ttl=5, show_spinner=False)
def fetch_health():
    try:
        r = requests.get(f"{API_BASE_URL}/health", timeout=3)
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass
    return None

@st.cache_data(ttl=60, show_spinner=False)
def fetch_model_info():
    try:
        r = requests.get(f"{API_BASE_URL}/model-info", timeout=5)
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass
    return None

@st.cache_data(ttl=60, show_spinner=False)
def fetch_feature_importance():
    try:
        r = requests.get(f"{API_BASE_URL}/feature-importance", timeout=5)
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass
    return []

def fetch_history(limit=50):
    try:
        r = requests.get(f"{API_BASE_URL}/history", params={"limit": limit}, timeout=6)
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass
    return []

def send_prediction(payload):
    r = requests.post(f"{API_BASE_URL}/predict", json=payload, timeout=10)
    r.raise_for_status()
    return r.json()

# =========================================================
# 4. SIDEBAR NAVIGATION
# =========================================================
health_data = fetch_health()
api_online = health_data is not None

with st.sidebar:
    st.markdown(
        """
        <div style="display: flex; align-items: center; gap: 12px; margin-bottom: 18px;">
            <div style="background: linear-gradient(135deg, #0284C7, #38BDF8); width: 40px; height: 40px; border-radius: 10px; display: flex; align-items: center; justify-content: center; font-size: 20px;">
                ⚡
            </div>
            <div>
                <div style="color: #FFFFFF; font-weight: 800; font-size: 18px; line-height: 1.1;">MACHINEGUARD</div>
                <div style="color: #38BDF8; font-size: 11px; font-weight: 600; letter-spacing: 0.05em;">AI TELEMETRY HUB</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("<div style='height: 1px; background: #1E293B; margin: 15px 0;'></div>", unsafe_allow_html=True)
    st.markdown("<p style='font-size: 11px; font-weight: 700; color: #64748B; letter-spacing: 0.1em; text-transform: uppercase;'>Workspaces</p>", unsafe_allow_html=True)

    page = st.radio(
        "Navigation",
        [
            "Mission Control",
            "Live Machine Assessment",
            "Telemetry & History",
            "Model Diagnostics",
        ],
        label_visibility="collapsed",
        key="navigation_view",
    )

    st.markdown("<div style='height: 1px; background: #1E293B; margin: 20px 0;'></div>", unsafe_allow_html=True)

    # API Connection Card
    if api_online:
        st.markdown(
            f"""
            <div style="background: rgba(16, 185, 129, 0.1); border: 1px solid rgba(16, 185, 129, 0.3); border-radius: 10px; padding: 12px;">
                <div style="display: flex; align-items: center; gap: 8px;">
                    <div style="width: 8px; height: 8px; border-radius: 50%; background: #10B981; box-shadow: 0 0 8px #10B981;"></div>
                    <span style="color: #10B981; font-weight: 700; font-size: 12px;">BACKEND ONLINE</span>
                </div>
                <div style="font-size: 11px; color: #94A3B8; margin-top: 4px;">
                    Model: <strong style="color: #E2E8F0;">{health_data.get('selected_model', 'XGBoost')}</strong><br/>
                    Threshold: <strong style="color: #E2E8F0;">{health_data.get('threshold', 0.84):.0%}</strong>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"""
            <div style="background: rgba(244, 63, 94, 0.1); border: 1px solid rgba(244, 63, 94, 0.3); border-radius: 10px; padding: 12px;">
                <div style="display: flex; align-items: center; gap: 8px;">
                    <div style="width: 8px; height: 8px; border-radius: 50%; background: #F43F5E;"></div>
                    <span style="color: #F43F5E; font-weight: 700; font-size: 12px;">BACKEND OFFLINE</span>
                </div>
                <div style="font-size: 11px; color: #94A3B8; margin-top: 4px;">
                    Endpoint: <code>{API_BASE_URL}</code><br/>
                    Start backend: <code>uvicorn backend.main:app</code>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown(
        """
        <div style="position: absolute; bottom: 20px; font-size: 11px; color: #475569;">
            System Version 2.0.0 Pro<br/>
            Clean Architecture Separation
        </div>
        """,
        unsafe_allow_html=True,
    )

# =========================================================
# 5. GLOBAL HERO BANNER
# =========================================================
st.markdown(
    """
    <div class="hero-container">
        <div class="hero-badge">⚡ Industrial Edge Intelligence</div>
        <h1 class="hero-title">Predictive Maintenance Operation Center</h1>
        <p class="hero-desc">Real-time industrial sensor diagnostics, XGBoost machine failure forecasting, and automated risk triage dispatch.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# =========================================================
# 6. VIEW 1: MISSION CONTROL (OVERVIEW)
# =========================================================
if page == "Mission Control":
    history = fetch_history(limit=500) if api_online else []
    total_preds = len(history)
    critical_alerts = sum(1 for r in history if r.get("predicted_failure") == 1 or r.get("status") == "FAILURE_RISK")
    nominal_runs = total_preds - critical_alerts
    avg_probability = (
        sum(float(r.get("failure_probability", 0)) for r in history) / total_preds
        if total_preds > 0
        else 0.0
    )

    # Top KPI Metrics
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Assessments Run", f"{total_preds:,}")
    c2.metric("Critical Alerts", f"{critical_alerts:,}", delta=f"{critical_alerts} machines" if critical_alerts else None, delta_color="inverse")
    c3.metric("Normal Operation", f"{nominal_runs:,}")
    c4.metric("Avg Predicted Risk", f"{avg_probability:.1%}")

    st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)

    # Interactive Dashboard Row
    col_chart, col_quick = st.columns([2, 1])

    with col_chart:
        st.markdown("### Risk Probability Distribution (Recent Records)")
        if history:
            df_hist = pd.DataFrame(history)
            fig = px.histogram(
                df_hist,
                x="failure_probability",
                nbins=20,
                color="status",
                color_discrete_map={
                    "FAILURE_RISK": "#F43F5E",
                    "WARNING_RISK": "#F59E0B",
                    "NORMAL_RISK": "#10B981",
                },
                template="plotly_dark",
            )
            fig.update_layout(
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                margin=dict(l=20, r=20, t=20, b=20),
                height=260,
                font=dict(color="#94A3B8"),
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No prediction telemetry available yet. Perform assessments to render live charts.")

    with col_quick:
        st.markdown("### Quick Diagnostic Actions")
        with st.container(border=True):
            st.markdown("#### 🔍 Run Sensor Inference")
            st.caption("Input rotational speed, temperatures, torque, and tool wear to trigger ML prediction.")
            if st.button("Open Assessment Form →", use_container_width=True):
                st.session_state["navigation_view"] = "Live Machine Assessment"
                st.rerun()

        with st.container(border=True):
            st.markdown("#### 📊 Model Diagnostics")
            st.caption("Inspect XGBoost decision threshold (84%), confusion matrix, and SHAP weights.")
            if st.button("Inspect Model Metrics →", use_container_width=True):
                st.session_state["navigation_view"] = "Model Diagnostics"
                st.rerun()

    # Recent Telemetry Log Preview
    if history:
        st.markdown("### Recent Log Telemetry")
        preview_df = pd.DataFrame(history[:6])
        cols = [c for c in ["id", "created_at", "machine_type", "torque_nm", "rotational_speed_rpm", "failure_probability", "status"] if c in preview_df.columns]
        st.dataframe(preview_df[cols], use_container_width=True, hide_index=True)

# =========================================================
# 7. VIEW 2: LIVE MACHINE ASSESSMENT
# =========================================================
elif page == "Live Machine Assessment":
    st.markdown("### 🛠️ Sensor Reading Input & Machine Inference")
    st.caption("Supply operational sensor parameters to compute real-time failure likelihood using the tuned XGBoost model.")

    with st.form("machine_inference_form"):
        # Machine Profile
        st.markdown("#### 1. Equipment Variant Profile")
        col_type, col_preset = st.columns([2, 2])
        with col_type:
            machine_type = st.selectbox(
                "Product Quality Variant",
                options=["L", "M", "H"],
                format_func=lambda x: {
                    "L": "L (Low Variant - 50% of volume)",
                    "M": "M (Medium Variant - 30% of volume)",
                    "H": "H (High Variant - 20% of volume)",
                }[x],
            )
        with col_preset:
            st.markdown("<p style='font-size: 13px; color: #94A3B8; margin-bottom: 6px;'>Quick Test Configurations</p>", unsafe_allow_html=True)
            preset = st.selectbox(
                "Load Preset Scenario",
                ["Standard Nominal Load", "Severe Tool Wear Overload", "Thermal & Torque Strain"],
                label_visibility="collapsed",
            )

        # Apply preset defaults
        if preset == "Standard Nominal Load":
            default_air, default_proc, default_rpm, default_torque, default_wear = 300.0, 310.0, 1500, 40.0, 40
        elif preset == "Severe Tool Wear Overload":
            default_air, default_proc, default_rpm, default_torque, default_wear = 302.5, 311.8, 1380, 58.0, 215
        else:
            default_air, default_proc, default_rpm, default_torque, default_wear = 304.5, 314.0, 1250, 65.5, 180

        st.markdown("<div style='height: 1px; background: #1E293B; margin: 15px 0;'></div>", unsafe_allow_html=True)
        st.markdown("#### 2. Physical Telemetry Measurements")

        col_left, col_right = st.columns(2)
        with col_left:
            air_temp = st.number_input(
                "Air Temperature [K]",
                min_value=250.0,
                max_value=350.0,
                value=float(default_air),
                step=0.1,
                help="Ambient environment temperature in Kelvin (Standard: 298-304K)",
            )
            rot_speed = st.number_input(
                "Rotational Speed [RPM]",
                min_value=1,
                max_value=10000,
                value=int(default_rpm),
                step=10,
                help="Spindle rotational velocity in RPM (Standard: 1300-1800 RPM)",
            )
            tool_wear = st.number_input(
                "Tool Wear Contact [min]",
                min_value=0,
                max_value=1000,
                value=int(default_wear),
                step=1,
                help="Accumulated cutting tool time. Replaced periodically.",
            )
        with col_right:
            proc_temp = st.number_input(
                "Process Temperature [K]",
                min_value=250.0,
                max_value=400.0,
                value=float(default_proc),
                step=0.1,
                help="Machine process operational temperature in Kelvin",
            )
            torque = st.number_input(
                "Applied Torque [Nm]",
                min_value=0.0,
                max_value=500.0,
                value=float(default_torque),
                step=0.5,
                help="Rotational force in Newton-metres (Standard: 30-50 Nm)",
            )

        submit_btn = st.form_submit_button("⚡ Execute AI Diagnostic Assessment", use_container_width=True)

    if submit_btn:
        if not api_online:
            st.error("Cannot perform inference: The FastAPI backend service is offline.")
        else:
            payload = {
                "type": machine_type,
                "air_temperature_k": air_temp,
                "process_temperature_k": proc_temp,
                "rotational_speed_rpm": rot_speed,
                "torque_nm": torque,
                "tool_wear_min": tool_wear,
            }
            try:
                with st.spinner("Analyzing operational vectors with XGBoost inference engine..."):
                    result = send_prediction(payload)
                st.session_state["latest_res"] = result
                st.session_state["latest_in"] = payload
                st.toast("Assessment completed!", icon="✅")
            except Exception as e:
                st.error(f"Prediction failed: {e}")

    # Display results
    if "latest_res" in st.session_state:
        res = st.session_state["latest_res"]
        prob = float(res.get("failure_probability", 0.0))
        thresh = float(res.get("decision_threshold", 0.84))
        is_fail = res.get("predicted_failure") == 1
        status = res.get("status", "NORMAL_RISK")
        risk_lvl = res.get("risk_level", "Nominal")

        if is_fail:
            card_class = "risk-critical"
            badge_color = "#F43F5E"
            status_icon = "🚨"
            status_title = "CRITICAL FAILURE RISK DETECTED"
        elif prob >= 0.40:
            card_class = "risk-warning"
            badge_color = "#F59E0B"
            status_icon = "⚠️"
            status_title = "ELEVATED RISK - PREVENTATIVE INSPECTION ADVISED"
        else:
            card_class = "risk-nominal"
            badge_color = "#10B981"
            status_icon = "🛡️"
            status_title = "NOMINAL OPERATING CONDITION"

        st.markdown(
            f"""
            <div class="risk-badge-card {card_class}">
                <div class="risk-status-tag" style="color: {badge_color};">{risk_lvl} Status · Decision Threshold: {thresh:.0%}</div>
                <div class="risk-title-val">{status_icon} {status_title}</div>
                <div class="risk-detail-text">{res.get('recommendation')}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        r1, r2, r3 = st.columns(3)
        r1.metric("Failure Probability", f"{prob:.1%}")
        r2.metric("Optimal Model Threshold", f"{thresh:.0%}")
        r3.metric("Operating Margin", f"{abs(prob - thresh):.1%}")

        # Gauge / Progress Visualizer
        st.markdown("<p style='font-size: 12px; color: #94A3B8; margin-top: 10px; margin-bottom: 2px;'>Risk Gradient Scale</p>", unsafe_allow_html=True)
        st.progress(min(1.0, prob))

# =========================================================
# 8. VIEW 3: TELEMETRY & HISTORY
# =========================================================
elif page == "Telemetry & History":
    st.markdown("### 🗄️ Historical Assessment Telemetry")
    st.caption("Review all recorded machine health predictions stored persistently in SQLite.")

    if not api_online:
        st.warning("FastAPI service is offline. Reconnect the backend to retrieve SQLite historical records.")
    else:
        fc1, fc2 = st.columns([3, 1])
        with fc1:
            record_limit = st.selectbox("Record Limit", [25, 50, 100, 250, 500], index=1)
        with fc2:
            st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
            if st.button("🔄 Refresh Data", use_container_width=True):
                st.rerun()

        records = fetch_history(limit=record_limit)
        if records:
            df = pd.DataFrame(records)
            st.dataframe(df, use_container_width=True, hide_index=True)

            csv = df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "📥 Export History as CSV",
                data=csv,
                file_name=f"predictive_maintenance_history_{datetime.now():%Y%m%d_%H%M}.csv",
                mime="text/csv",
            )
        else:
            st.info("No records recorded yet in SQLite database.")

# =========================================================
# 9. VIEW 4: MODEL DIAGNOSTICS & SHAP
# =========================================================
elif page == "Model Diagnostics":
    st.markdown("### 🧠 Model Performance & Explainability (SHAP)")
    st.caption("Deep-dive into the production XGBoost classifier, cross-validation metrics, and global feature impacts.")

    model_info = fetch_model_info()
    shap_data = fetch_feature_importance()

    if model_info:
        tm = model_info.get("test_metrics", {})
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("ROC-AUC", f"{tm.get('roc_auc', 0):.3f}")
        m2.metric("PR-AUC", f"{tm.get('pr_auc', 0):.3f}")
        m3.metric("F1-Score", f"{tm.get('f1_score', 0):.3f}")
        m4.metric("Test Accuracy", f"{tm.get('accuracy', 0):.1%}")

        st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)

        conf = tm.get("confusion_matrix", {})
        if conf:
            st.markdown("#### Test Set Confusion Matrix (2,000 Test Records)")
            cm_col1, cm_col2, cm_col3, cm_col4 = st.columns(4)
            cm_col1.metric("True Negatives", f"{conf.get('true_negatives'):,}")
            cm_col2.metric("False Positives", f"{conf.get('false_positives'):,}")
            cm_col3.metric("False Negatives", f"{conf.get('false_negatives'):,}")
            cm_col4.metric("True Positives", f"{conf.get('true_positives'):,}")

    if shap_data:
        st.markdown("#### Global Feature Importance (Mean Absolute SHAP Value)")
        df_shap = pd.DataFrame(shap_data)
        fig_shap = px.bar(
            df_shap,
            x="mean_absolute_shap",
            y="feature",
            orientation="h",
            color="mean_absolute_shap",
            color_continuous_scale="Blues",
            template="plotly_dark",
        )
        fig_shap.update_layout(
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=20, r=20, t=10, b=10),
            height=320,
            yaxis=dict(autorange="reversed"),
        )
        st.plotly_chart(fig_shap, use_container_width=True)

# =========================================================
# 10. GLOBAL FOOTER
# =========================================================
st.markdown(
    """
    <div class="pro-footer">
        MachineGuard Pro · Clean Architecture (FastAPI Backend + Streamlit UI Frontend) · Industrial ML Lab
    </div>
    """,
    unsafe_allow_html=True,
)
