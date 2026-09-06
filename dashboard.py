import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="EPBL Smart STP",
    page_icon="💧",
    layout="wide"
)


# ============================================================
# LOAD DATA
# ============================================================

DATA_PATH = "data/stp_sensor_data.csv"
RISK_PATH = "outputs/metrics/risk_engine_results.csv"

df = pd.read_csv(DATA_PATH)
risk_df = pd.read_csv(RISK_PATH)

df["timestamp"] = pd.to_datetime(df["timestamp"])
risk_df["timestamp"] = pd.to_datetime(risk_df["timestamp"])


# ============================================================
# HEADER
# ============================================================

st.title("💧 EPBL Smart STP")
st.subheader(
    "AI-Powered Predictive Monitoring & Anomaly Detection System"
)

st.caption(
    "Deep Learning prototype using synthetic STP operational data"
)

st.divider()


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header("Dashboard Controls")

date_range = st.sidebar.date_input(
    "Select date range",
    value=(
        df["timestamp"].min().date(),
        df["timestamp"].max().date()
    )
)

risk_filter = st.sidebar.multiselect(
    "Risk level",
    ["LOW", "MEDIUM", "HIGH"],
    default=["LOW", "MEDIUM", "HIGH"]
)


# ============================================================
# FILTER DATA
# ============================================================

if len(date_range) == 2:

    start_date = pd.Timestamp(date_range[0])

    end_date = (
        pd.Timestamp(date_range[1])
        + pd.Timedelta(days=1)
    )

    filtered_risk = risk_df[
        (risk_df["timestamp"] >= start_date)
        &
        (risk_df["timestamp"] < end_date)
        &
        (risk_df["risk_level"].isin(risk_filter))
    ].copy()

else:

    filtered_risk = risk_df[
        risk_df["risk_level"].isin(risk_filter)
    ].copy()


# ============================================================
# CURRENT STATUS
# ============================================================

latest = risk_df.iloc[-1]


current_do = latest["current_DO"]
forecast_do = latest["forecast_DO"]

current_ph = latest["pH"]

current_flow = latest["flow_rate"]

current_energy = latest["energy_consumption"]

current_anomaly = latest["anomaly_detected"]

current_risk = latest["risk_level"]

risk_score = latest["risk_score"]

reconstruction_error = latest[
    "reconstruction_error"
]


# ============================================================
# KPI CARDS
# ============================================================

st.subheader("Current Plant Status")

col1, col2, col3, col4, col5 = st.columns(5)


with col1:

    st.metric(
        "Current DO",
        f"{current_do:.2f} mg/L"
    )


with col2:

    st.metric(
        "Next-Hour DO Forecast",
        f"{forecast_do:.2f} mg/L"
    )


with col3:

    if current_anomaly == 1:

        st.metric(
            "AI Anomaly",
            "DETECTED"
        )

    else:

        st.metric(
            "AI Anomaly",
            "NORMAL"
        )


with col4:

    st.metric(
        "Risk Score",
        f"{risk_score}"
    )


with col5:

    if current_risk == "HIGH":

        st.error(
            "🔴 HIGH RISK"
        )

    elif current_risk == "MEDIUM":

        st.warning(
            "🟠 MEDIUM RISK"
        )

    else:

        st.success(
            "🟢 LOW RISK"
        )


st.divider()


# ============================================================
# CURRENT PARAMETERS
# ============================================================

st.subheader("Current STP Parameters")

p1, p2, p3, p4 = st.columns(4)

with p1:

    st.metric(
        "pH",
        f"{current_ph:.2f}"
    )

with p2:

    st.metric(
        "Flow Rate",
        f"{current_flow:.0f}"
    )

with p3:

    st.metric(
        "Energy Consumption",
        f"{current_energy:.1f}"
    )

with p4:

    st.metric(
        "Reconstruction Error",
        f"{reconstruction_error:.3f}"
    )


st.divider()


# ============================================================
# DO TREND + FORECAST
# ============================================================

st.subheader("Dissolved Oxygen Monitoring")

recent = risk_df.tail(168).copy()

fig_do = go.Figure()

fig_do.add_trace(
    go.Scatter(
        x=recent["timestamp"],
        y=recent["current_DO"],
        mode="lines",
        name="Actual DO"
    )
)

fig_do.add_trace(
    go.Scatter(
        x=recent["timestamp"],
        y=recent["forecast_DO"],
        mode="lines",
        name="LSTM Forecast",
        line=dict(
            dash="dash"
        )
    )
)

fig_do.add_hline(
    y=2.0,
    line_dash="dot",
    annotation_text="DO warning level"
)

fig_do.update_layout(
    height=450,
    xaxis_title="Time",
    yaxis_title="Dissolved Oxygen (mg/L)",
    hovermode="x unified"
)

st.plotly_chart(
    fig_do,
    use_container_width=True
)


# ============================================================
# ANOMALY DETECTION
# ============================================================

st.subheader("AI Anomaly Detection")

fig_anomaly = go.Figure()

fig_anomaly.add_trace(
    go.Scatter(
        x=recent["timestamp"],
        y=recent["reconstruction_error"],
        mode="lines",
        name="Reconstruction Error"
    )
)

fig_anomaly.add_hline(
    y=0.456434,
    line_dash="dash",
    annotation_text="AI anomaly threshold"
)

anomaly_points = recent[
    recent["anomaly_detected"] == 1
]

fig_anomaly.add_trace(
    go.Scatter(
        x=anomaly_points["timestamp"],
        y=anomaly_points["reconstruction_error"],
        mode="markers",
        name="Detected Anomaly",
        marker=dict(
            size=9
        )
    )
)

fig_anomaly.update_layout(
    height=450,
    xaxis_title="Time",
    yaxis_title="Reconstruction Error",
    hovermode="x unified"
)

st.plotly_chart(
    fig_anomaly,
    use_container_width=True
)


# ============================================================
# RISK DISTRIBUTION
# ============================================================

st.subheader("Risk Distribution")

risk_counts = (
    filtered_risk["risk_level"]
    .value_counts()
    .reindex(
        ["LOW", "MEDIUM", "HIGH"],
        fill_value=0
    )
)

fig_risk = go.Figure(
    data=[
        go.Bar(
            x=risk_counts.index,
            y=risk_counts.values,
            text=risk_counts.values,
            textposition="auto"
        )
    ]
)

fig_risk.update_layout(
    height=350,
    xaxis_title="Risk Level",
    yaxis_title="Number of Observations"
)

st.plotly_chart(
    fig_risk,
    use_container_width=True
)


# ============================================================
# RISK TIMELINE
# ============================================================

st.subheader("AI Risk Timeline")

recent_risk = filtered_risk.tail(168)

risk_numeric = {
    "LOW": 1,
    "MEDIUM": 2,
    "HIGH": 3
}

risk_values = (
    recent_risk["risk_level"]
    .map(risk_numeric)
)

fig_risk_timeline = go.Figure()

fig_risk_timeline.add_trace(
    go.Scatter(
        x=recent_risk["timestamp"],
        y=risk_values,
        mode="lines+markers",
        name="Risk Level"
    )
)

fig_risk_timeline.update_layout(
    height=350,
    xaxis_title="Time",
    yaxis_title="Risk",
    yaxis=dict(
        tickmode="array",
        tickvals=[1, 2, 3],
        ticktext=["LOW", "MEDIUM", "HIGH"]
    )
)

st.plotly_chart(
    fig_risk_timeline,
    use_container_width=True
)


# ============================================================
# HIGH-RISK EVENTS
# ============================================================

st.subheader("🚨 High-Risk Events")

high_risk = filtered_risk[
    filtered_risk["risk_level"] == "HIGH"
].copy()

if len(high_risk) > 0:

    display_columns = [
        "timestamp",
        "current_DO",
        "forecast_DO",
        "reconstruction_error",
        "risk_score",
        "risk_reasons",
        "recommendation"
    ]

    st.dataframe(
        high_risk[
            display_columns
        ].sort_values(
            "timestamp",
            ascending=False
        ).head(20),
        use_container_width=True,
        hide_index=True
    )

else:

    st.success(
        "No high-risk events in the selected period."
    )


# ============================================================
# AI RECOMMENDATION
# ============================================================

st.divider()

st.subheader("🤖 AI Operator Recommendation")

recommendation = latest["recommendation"]

if current_risk == "HIGH":

    st.error(
        f"🔴 {recommendation}"
    )

elif current_risk == "MEDIUM":

    st.warning(
        f"🟠 {recommendation}"
    )

else:

    st.success(
        f"🟢 {recommendation}"
    )


# ============================================================
# SYSTEM EXPLANATION
# ============================================================

with st.expander(
    "How does the AI system work?"
):

    st.markdown(
        """
        ### EPBL Smart STP AI Pipeline

        **1. Sensor Data**

        The system receives simulated STP operational
        parameters such as pH, dissolved oxygen,
        flow rate, COD, BOD, TSS, energy consumption,
        pump condition and aeration condition.

        **2. LSTM Forecasting**

        The LSTM model analyses the previous 24 hours
        of process behaviour and forecasts the next-hour
        dissolved oxygen level.

        **3. LSTM Autoencoder**

        The autoencoder learns normal STP behaviour.
        A high reconstruction error indicates behaviour
        that differs significantly from normal operation.

        **4. AI Risk Engine**

        Forecasts, anomaly signals and current operating
        conditions are combined into a risk score.

        **5. Operator Recommendation**

        The system produces LOW, MEDIUM or HIGH risk
        and provides a recommended operator response.

        ---
        
        **Important:** This prototype uses synthetic data
        because publicly available EPBL operational sensor
        data is not available. The architecture can be
        retrained and calibrated using real plant data
        when available.
        """
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "EPBL Smart STP | Deep Learning Predictive Monitoring Prototype"
)