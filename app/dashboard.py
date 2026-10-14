
import os
from datetime import datetime

import pandas as pd
import requests
import streamlit as st


# =========================================================
# CONFIGURATION
# =========================================================
API_BASE_URL = os.getenv(
    "PREDICTIVE_MAINTENANCE_API",
    "http://127.0.0.1:8000",
).rstrip("/")

st.set_page_config(
    page_title="MachineGuard | Predictive Maintenance",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =========================================================
# CUSTOM THEME
# =========================================================
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');

    :root {
        --navy: #10243A;
        --navy-light: #183651;
        --teal: #087F78;
        --teal-light: #0F9D91;
        --background: #F4F7FB;
        --white: #FFFFFF;
        --text: #1D2D3F;
        --muted: #64748B;
        --border: #DFE7F0;
        --red: #C43D3D;
    }

    /* Main application */
    .stApp {
        background-color: var(--background) !important;
        color: var(--text) !important;
        font-family: 'DM Sans', sans-serif !important;
    }

    /* Main page text */
    .stApp h1,
    .stApp h2,
    .stApp h3,
    .stApp h4 {
        color: var(--navy) !important;
        font-family: 'Space Grotesk', sans-serif !important;
    }

    .stApp p,
    .stApp li {
        color: var(--text);
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: var(--navy) !important;
    }

    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3,
    [data-testid="stSidebar"] h4,
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] {
        color: #F4F8FC !important;
    }

    [data-testid="stSidebar"] [data-testid="stCaptionContainer"],
    [data-testid="stSidebar"] [data-testid="stCaptionContainer"] p {
        color: #C6D5E4 !important;
    }

    [data-testid="stSidebar"] hr {
        border-color: #355069 !important;
    }

    /* Header banner */
    .hero {
        background: linear-gradient(
            120deg,
            #10243A 0%,
            #183F55 65%,
            #087F78 100%
        );
        padding: 30px 32px;
        border-radius: 20px;
        margin-bottom: 25px;
        box-shadow: 0 10px 28px rgba(16, 36, 58, 0.12);
    }

    .hero .eyebrow {
        color: #80E1D4 !important;
        text-transform: uppercase;
        letter-spacing: 2px;
        font-size: 11px;
        font-weight: 700;
    }

    .hero h1 {
        color: #FFFFFF !important;
        font-size: 31px;
        margin: 7px 0 8px 0;
        letter-spacing: -0.7px;
    }

    .hero p {
        color: #E5EEF5 !important;
        margin: 0;
        font-size: 14px;
    }

    .section-heading {
        font-family: 'Space Grotesk', sans-serif;
        color: #10243A !important;
        font-size: 22px;
        font-weight: 700;
        margin: 10px 0 4px 0;
    }

    .section-caption {
        color: #64748B !important;
        font-size: 13px;
        margin-bottom: 18px;
    }

    /* Form labels */
    .stApp [data-testid="stWidgetLabel"],
    .stApp [data-testid="stWidgetLabel"] p,
    .stApp [data-testid="stWidgetLabel"] label,
    .stApp [data-testid="stNumberInput"] label,
    .stApp [data-testid="stSelectbox"] label,
    .stApp [data-testid="stRadio"] label {
        color: #10243A !important;
        opacity: 1 !important;
        font-weight: 600 !important;
    }

    /* Explicit measurement headings */
    .stApp [data-testid="stMarkdownContainer"] strong {
        color: #10243A !important;
        font-weight: 700 !important;
    }

    /* Helper text */
    .stApp [data-testid="stCaptionContainer"],
    .stApp [data-testid="stCaptionContainer"] p {
        color: #64748B !important;
        opacity: 1 !important;
        font-size: 13px !important;
        line-height: 1.5 !important;
    }

    /* Number input fields */
    .stApp [data-testid="stNumberInput"] input {
        color: #1D2D3F !important;
        -webkit-text-fill-color: #1D2D3F !important;
        background-color: #FFFFFF !important;
        opacity: 1 !important;
    }

    .stApp [data-testid="stNumberInput"] button {
        color: #10243A !important;
    }

    .stApp [data-testid="stNumberInput"] button svg {
        fill: #10243A !important;
        stroke: #10243A !important;
    }

    /* Select box */
    .stApp [data-baseweb="select"] > div {
        background-color: #FFFFFF !important;
        border-color: #CFD9E5 !important;
        color: #1D2D3F !important;
    }

    .stApp [data-baseweb="select"] input {
        color: #1D2D3F !important;
        -webkit-text-fill-color: #1D2D3F !important;
    }

    /* Dropdown options */
    [data-baseweb="popover"],
    [data-baseweb="menu"] {
        background-color: #FFFFFF !important;
    }

    [role="option"] {
        color: #1D2D3F !important;
        background-color: #FFFFFF !important;
    }

    [role="option"]:hover {
        background-color: #E3F7F3 !important;
    }

    /* Buttons: explicit foreground and background */
    .stApp div.stButton > button,
    .stApp div.stFormSubmitButton > button,
    .stApp div.stDownloadButton > button {
        background-color: #087F78 !important;
        color: #FFFFFF !important;
        border: 1px solid #087F78 !important;
        border-radius: 10px !important;
        min-height: 44px;
        font-weight: 700 !important;
        opacity: 1 !important;
    }

    .stApp div.stButton > button *,
    .stApp div.stFormSubmitButton > button *,
    .stApp div.stDownloadButton > button * {
        color: #FFFFFF !important;
        -webkit-text-fill-color: #FFFFFF !important;
        opacity: 1 !important;
    }

    .stApp div.stButton > button:hover,
    .stApp div.stFormSubmitButton > button:hover,
    .stApp div.stDownloadButton > button:hover {
        background-color: #066A64 !important;
        border-color: #066A64 !important;
        color: #FFFFFF !important;
    }

    .stApp div.stButton > button:disabled,
    .stApp div.stFormSubmitButton > button:disabled {
        background-color: #A9B9C7 !important;
        color: #FFFFFF !important;
        opacity: 1 !important;
    }

    /* Metric cards */
    .stApp [data-testid="stMetric"] {
        background-color: #FFFFFF !important;
        border: 1px solid #DFE7F0 !important;
        border-radius: 14px !important;
        padding: 18px !important;
    }

    .stApp [data-testid="stMetricLabel"],
    .stApp [data-testid="stMetricLabel"] p {
        color: #64748B !important;
    }

    .stApp [data-testid="stMetricValue"],
    .stApp [data-testid="stMetricValue"] div {
        color: #10243A !important;
    }

    /* Result cards */
    .result-card {
        background-color: #FFFFFF;
        border: 1px solid #DFE7F0;
        border-radius: 18px;
        padding: 25px;
        margin: 12px 0 20px 0;
    }

    .result-safe {
        border-top: 5px solid #0F9D91;
    }

    .result-alert {
        border-top: 5px solid #C43D3D;
    }

    .result-label {
        color: #64748B !important;
        font-size: 12px;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 1.2px;
    }

    .result-value {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 28px;
        font-weight: 700;
        margin: 8px 0;
    }

    .result-description {
        color: #64748B !important;
        font-size: 13px;
        line-height: 1.6;
    }

    /* Expanders */
    .stApp [data-testid="stExpander"] {
        background-color: #FFFFFF !important;
        border: 1px solid #DFE7F0 !important;
        border-radius: 12px !important;
    }

    .stApp [data-testid="stExpander"] summary,
    .stApp [data-testid="stExpander"] summary p {
        color: #10243A !important;
    }

    /* Dataframes */
    .stApp [data-testid="stDataFrame"] {
        border: 1px solid #DFE7F0;
        border-radius: 12px;
        overflow: hidden;
    }

    /* Footer */
    .footer {
        text-align: center;
        color: #64748B !important;
        font-size: 12px;
        padding: 24px 0 10px 0;
    }

    #MainMenu,
    footer {
        visibility: hidden;
    }

    @media (max-width: 768px) {
        .hero {
            padding: 22px;
        }

        .hero h1 {
            font-size: 25px;
        }

        .result-value {
            font-size: 23px;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# API HELPERS
# =========================================================
@st.cache_data(ttl=10, show_spinner=False)
def check_api_health():
    try:
        response = requests.get(
            f"{API_BASE_URL}/health",
            timeout=4,
        )
        response.raise_for_status()
        return True
    except requests.RequestException:
        return False


def get_history(limit=50):
    response = requests.get(
        f"{API_BASE_URL}/history",
        params={"limit": limit},
        timeout=10,
    )
    response.raise_for_status()

    data = response.json()

    if isinstance(data, list):
        return data

    if isinstance(data, dict):
        for key in ("history", "items", "records", "predictions"):
            if isinstance(data.get(key), list):
                return data[key]

    return []


def get_prediction(payload):
    response = requests.post(
        f"{API_BASE_URL}/predict",
        json=payload,
        timeout=20,
    )
    response.raise_for_status()
    return response.json()


def first_value(data, *keys, default=None):
    for key in keys:
        if key in data and data[key] is not None:
            return data[key]
    return default


def to_probability(value):
    """Convert a probability or percentage to a 0–1 value."""
    number = float(value)

    if number > 1:
        number /= 100

    return max(0.0, min(1.0, number))


def normalize_failure_prediction(result):
    value = first_value(
        result,
        "predicted_failure",
        "prediction",
        default=0,
    )

    if isinstance(value, str):
        return value.strip().lower() in (
            "1", "true", "yes", "failure", "alert"
        )

    return bool(value)


def format_probability(value):
    try:
        number = float(value)
        if number <= 1:
            return f"{number:.1%}"
        return f"{number:.1f}%"
    except (TypeError, ValueError):
        return "—"


# =========================================================
# SIDEBAR
# =========================================================
api_online = check_api_health()

with st.sidebar:
    st.markdown(
        """
        <div style="padding: 12px 0 20px 0;">
            <div style="font-size: 30px;">⚙️</div>
            <div style="font-size: 20px; font-weight: 700;
                        font-family: 'Space Grotesk', sans-serif;">
                MachineGuard
            </div>
            <div style="font-size: 12px; color: #C6D5E4;">
                Predictive Maintenance
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("---")
    st.markdown("**CONTROL CENTER**")

    page = st.radio(
        "Navigation",
        [
            "Overview",
            "New Prediction",
            "Prediction History",
        ],
        label_visibility="collapsed",
        key="navigation",
    )

    st.markdown("---")

    if api_online:
        st.success("API connected")
    else:
        st.error("API unavailable")
        st.caption(
            "Start the FastAPI server to enable predictions."
        )

    st.markdown("---")
    st.caption("ML-assisted machine failure risk monitoring")


# =========================================================
# PAGE HEADER
# =========================================================
st.markdown(
    """
    <div class="hero">
        <div class="eyebrow">Industrial intelligence platform</div>
        <h1>Predictive Maintenance Center</h1>
        <p>
            Monitor machine conditions, estimate failure risk,
            and support proactive maintenance decisions.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# OVERVIEW PAGE
# =========================================================
if page == "Overview":

    st.markdown(
        '<div class="section-heading">Operations overview</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="section-caption">'
        'Review prediction activity and system readiness.'
        '</div>',
        unsafe_allow_html=True,
    )

    history = []

    if api_online:
        try:
            history = get_history(500)
        except requests.RequestException:
            st.warning(
                "Prediction history could not be loaded. "
                "Check the API and try again."
            )

    total_predictions = len(history)

    failure_count = sum(
        normalize_failure_prediction(row)
        for row in history
    )

    normal_count = total_predictions - failure_count

    metric1, metric2, metric3 = st.columns(3)

    metric1.metric(
        "Recorded predictions",
        f"{total_predictions:,}",
    )

    metric2.metric(
        "Failure alerts",
        f"{failure_count:,}",
    )

    metric3.metric(
        "Normal predictions",
        f"{normal_count:,}",
    )

    st.markdown("### Quick actions")

    action1, action2 = st.columns(2)

    with action1:
        with st.container(border=True):
            st.markdown("#### 🧪 Assess a machine")
            st.write(
                "Enter operating measurements and estimate "
                "the machine's failure probability."
            )

            if st.button(
                "Start new prediction →",
                use_container_width=True,
                key="start_prediction",
            ):
                st.session_state["navigation"] = "New Prediction"
                st.rerun()

    with action2:
        with st.container(border=True):
            st.markdown("#### 🗂️ Prediction history")
            st.write(
                "Review previous assessments and export saved records."
            )

            if st.button(
                "View history →",
                use_container_width=True,
                key="open_history",
            ):
                st.session_state["navigation"] = "Prediction History"
                st.rerun()

    if not api_online:
        st.warning(
            "The dashboard is available, but live predictions "
            "and saved history require the FastAPI service."
        )

    if history:
        st.markdown("### Recent activity")

        recent_df = pd.DataFrame(history[:8])

        rename_map = {
            "created_at": "Recorded at",
            "machine_type": "Machine type",
            "type": "Machine type",
            "failure_probability": "Failure probability",
            "predicted_failure": "Predicted failure",
            "status": "Status",
        }

        recent_df = recent_df.rename(
            columns={
                key: value
                for key, value in rename_map.items()
                if key in recent_df.columns
            }
        )

        visible_columns = [
            "Recorded at",
            "Machine type",
            "Failure probability",
            "Predicted failure",
            "Status",
        ]

        visible_columns = [
            col for col in visible_columns
            if col in recent_df.columns
        ]

        if visible_columns:
            recent_df = recent_df[visible_columns]

        st.dataframe(
            recent_df,
            use_container_width=True,
            hide_index=True,
        )


# =========================================================
# NEW PREDICTION PAGE
# =========================================================
elif page == "New Prediction":

    st.markdown(
        '<div class="section-heading">Machine assessment</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="section-caption">'
        'Enter current machine readings to assess potential failure risk.'
        '</div>',
        unsafe_allow_html=True,
    )

    with st.container(border=True):

        st.markdown("### 01 / Machine profile")
        st.caption(
            "Choose the machine category used by the model."
        )

        with st.form("machine_prediction_form"):

            machine_type = st.selectbox(
                "Machine type",
                options=["L", "M", "H"],
                format_func=lambda value: {
                    "L": "L — Low quality variant",
                    "M": "M — Medium quality variant",
                    "H": "H — High quality variant",
                }[value],
                help="Select the category matching your machine.",
            )

            st.markdown("### 02 / Operating measurements")

            st.caption(
                "Enter current sensor readings. Examples explain each "
                "field; they are not recommended operating settings."
            )

            left, right = st.columns(2)

            with left:
                st.markdown("**Air temperature (K)**")
                st.caption(
                    "Ambient air temperature around the machine. "
                    "Example: 300 K."
                )

                air_temperature = st.number_input(
                    "Air temperature value",
                    min_value=250.0,
                    max_value=350.0,
                    value=300.0,
                    step=0.1,
                    format="%.1f",
                    label_visibility="collapsed",
                )

                st.markdown("**Rotational speed (RPM)**")
                st.caption(
                    "How fast the machine rotates. Example: 1500 RPM."
                )

                rotational_speed = st.number_input(
                    "Rotational speed value",
                    min_value=1,
                    max_value=10000,
                    value=1500,
                    step=10,
                    label_visibility="collapsed",
                )

                st.markdown("**Tool wear (minutes)**")
                st.caption(
                    "Accumulated tool usage time. Example: 120 minutes."
                )

                tool_wear = st.number_input(
                    "Tool wear value",
                    min_value=0,
                    max_value=1000,
                    value=0,
                    step=1,
                    label_visibility="collapsed",
                )

            with right:
                st.markdown("**Process temperature (K)**")
                st.caption(
                    "Temperature during machine operation. "
                    "Example: 310 K."
                )

                process_temperature = st.number_input(
                    "Process temperature value",
                    min_value=250.0,
                    max_value=400.0,
                    value=310.0,
                    step=0.1,
                    format="%.1f",
                    label_visibility="collapsed",
                )

                st.markdown("**Torque (Nm)**")
                st.caption(
                    "Rotational force applied by the machine. "
                    "Example: 40 Nm."
                )

                torque = st.number_input(
                    "Torque value",
                    min_value=0.0,
                    max_value=500.0,
                    value=40.0,
                    step=0.5,
                    format="%.1f",
                    label_visibility="collapsed",
                )

            submitted = st.form_submit_button(
                "Analyze machine condition",
                use_container_width=True,
            )

    if submitted:

        if not api_online:
            st.error(
                "The prediction API is offline. "
                "Start FastAPI and try again."
            )

        else:
            payload = {
                "type": machine_type,
                "air_temperature_k": air_temperature,
                "process_temperature_k": process_temperature,
                "rotational_speed_rpm": rotational_speed,
                "torque_nm": torque,
                "tool_wear_min": tool_wear,
            }

            try:
                with st.spinner(
                    "Analyzing machine operating conditions..."
                ):
                    result = get_prediction(payload)

                st.session_state["latest_prediction"] = result
                st.session_state["latest_input"] = payload

                st.success("Assessment completed successfully.")

            except requests.HTTPError as exc:
                detail = (
                    "The API rejected the submitted measurements."
                )

                try:
                    detail = exc.response.json().get(
                        "detail",
                        detail,
                    )
                except (ValueError, AttributeError):
                    pass

                st.error(f"Prediction failed: {detail}")

            except requests.RequestException:
                st.error(
                    "Could not reach the prediction service. "
                    "Check the FastAPI server and API URL."
                )

    result = st.session_state.get("latest_prediction")

    if result:

        predicted_failure = normalize_failure_prediction(result)

        probability_raw = first_value(
            result,
            "failure_probability",
            "probability",
            "failure_probability_percent",
            default=0.0,
        )

        probability = to_probability(probability_raw)

        threshold_raw = first_value(
            result,
            "decision_threshold",
            "threshold",
            default=None,
        )

        status_text = (
            "Potential failure detected"
            if predicted_failure
            else "No failure predicted"
        )

        recommendation = first_value(
            result,
            "recommendation",
            default=(
                "Arrange a machine inspection and review its operating "
                "conditions."
                if predicted_failure
                else "Continue routine monitoring and follow the "
                "maintenance schedule."
            ),
        )

        card_class = (
            "result-alert"
            if predicted_failure
            else "result-safe"
        )

        status_color = (
            "#C43D3D"
            if predicted_failure
            else "#087F78"
        )

        status_icon = "⚠️" if predicted_failure else "✓"

        st.markdown("### Assessment result")

        st.markdown(
            f"""
            <div class="result-card {card_class}">
                <div class="result-label">Machine condition</div>
                <div class="result-value" style="color: {status_color};">
                    {status_icon} {status_text}
                </div>
                <div class="result-description">
                    This is a machine-learning estimate, not a guarantee
                    of the machine's actual condition.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        metric1, metric2 = st.columns(2)

        metric1.metric(
            "Estimated failure probability",
            f"{probability:.1%}",
        )

        metric2.metric(
            "Decision threshold",
            (
                f"{to_probability(threshold_raw):.1%}"
                if threshold_raw is not None
                else "Configured in model"
            ),
        )

        st.progress(probability)

        st.markdown("### Recommended next step")
        st.info(recommendation)

        with st.expander("Review submitted measurements"):

            input_data = st.session_state.get(
                "latest_input",
                {},
            )

            if input_data:
                readable = {
                    "Machine type": input_data.get("type"),
                    "Air temperature (K)": input_data.get(
                        "air_temperature_k"
                    ),
                    "Process temperature (K)": input_data.get(
                        "process_temperature_k"
                    ),
                    "Rotational speed (RPM)": input_data.get(
                        "rotational_speed_rpm"
                    ),
                    "Torque (Nm)": input_data.get("torque_nm"),
                    "Tool wear (minutes)": input_data.get(
                        "tool_wear_min"
                    ),
                }

                st.dataframe(
                    pd.DataFrame(
                        {
                            "Measurement": list(readable.keys()),
                            "Value": list(readable.values()),
                        }
                    ),
                    use_container_width=True,
                    hide_index=True,
                )


# =========================================================
# PREDICTION HISTORY PAGE
# =========================================================
elif page == "Prediction History":

    st.markdown(
        '<div class="section-heading">Prediction history</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="section-caption">'
        'Review saved machine assessments and export records.'
        '</div>',
        unsafe_allow_html=True,
    )

    if not api_online:
        st.warning(
            "Start the FastAPI service to retrieve saved predictions."
        )

    else:
        filter_col, refresh_col = st.columns([3, 1])

        with filter_col:
            limit = st.selectbox(
                "Records to display",
                options=[25, 50, 100, 250, 500],
                index=1,
            )

        with refresh_col:
            st.markdown(
                "<div style='height: 27px'></div>",
                unsafe_allow_html=True,
            )

            st.button(
                "Refresh history",
                use_container_width=True,
                on_click=lambda: st.session_state.update(
                    {"history_refresh": datetime.now().isoformat()}
                ),
            )

        try:
            history = get_history(limit)

            if not history:
                st.info(
                    "No predictions have been recorded yet. "
                    "Run your first assessment from New Prediction."
                )

            else:
                df = pd.DataFrame(history)

                rename_map = {
                    "id": "Record ID",
                    "created_at": "Recorded at",
                    "machine_type": "Machine type",
                    "type": "Machine type",
                    "air_temperature_k": "Air temp (K)",
                    "process_temperature_k": "Process temp (K)",
                    "rotational_speed_rpm": "Speed (RPM)",
                    "torque_nm": "Torque (Nm)",
                    "tool_wear_min": "Tool wear (min)",
                    "failure_probability": "Failure probability",
                    "decision_threshold": "Decision threshold",
                    "predicted_failure": "Predicted failure",
                    "status": "Status",
                }

                df = df.rename(
                    columns={
                        key: value
                        for key, value in rename_map.items()
                        if key in df.columns
                    }
                )

                for column in (
                    "Failure probability",
                    "Decision threshold",
                ):
                    if column in df.columns:
                        df[column] = df[column].map(
                            format_probability
                        )

                if "Predicted failure" in df.columns:
                    df["Predicted failure"] = (
                        df["Predicted failure"].map(
                            lambda value: (
                                "Alert"
                                if str(value).strip().lower()
                                in (
                                    "1", "true", "yes",
                                    "failure", "alert"
                                )
                                else "Normal"
                            )
                        )
                    )

                preferred_columns = [
                    "Record ID",
                    "Recorded at",
                    "Machine type",
                    "Air temp (K)",
                    "Process temp (K)",
                    "Speed (RPM)",
                    "Torque (Nm)",
                    "Tool wear (min)",
                    "Failure probability",
                    "Decision threshold",
                    "Predicted failure",
                    "Status",
                ]

                visible_columns = [
                    column
                    for column in preferred_columns
                    if column in df.columns
                ]

                if visible_columns:
                    df = df[visible_columns]

                st.dataframe(
                    df,
                    use_container_width=True,
                    hide_index=True,
                )

                st.caption(
                    f"Showing {len(df)} saved prediction(s)."
                )

                csv_data = df.to_csv(
                    index=False
                ).encode("utf-8")

                st.download_button(
                    "Export history as CSV",
                    data=csv_data,
                    file_name=(
                        f"prediction_history_"
                        f"{datetime.now():%Y%m%d_%H%M}.csv"
                    ),
                    mime="text/csv",
                )

        except requests.RequestException:
            st.error(
                "Could not load prediction history. "
                "Check the API and try again."
            )


# =========================================================
# FOOTER
# =========================================================
st.markdown(
    """
    <div class="footer">
        MachineGuard · Predictive Maintenance System ·
        ML-assisted decisions
    </div>
    """,
    unsafe_allow_html=True,
)
