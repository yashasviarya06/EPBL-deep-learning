import os
import numpy as np
import pandas as pd
import joblib

from tensorflow.keras.models import load_model


# ============================================================
# CONFIGURATION
# ============================================================

DATA_PATH = "data/stp_sensor_data.csv"

SCALER_PATH = "models/stp_scaler.pkl"

LSTM_MODEL_PATH = "models/lstm_stp_forecaster.keras"

AUTOENCODER_MODEL_PATH = "models/lstm_autoencoder.keras"

THRESHOLD_PATH = "outputs/metrics/anomaly_threshold.txt"

OUTPUT_PATH = "outputs/metrics/risk_engine_results.csv"

SEQUENCE_LENGTH = 24

BATCH_SIZE = 128

FEATURES = [
    "pH",
    "temperature",
    "DO",
    "TDS",
    "turbidity",
    "ORP",
    "flow_rate",
    "influent_load",
    "COD",
    "BOD",
    "TSS",
    "energy_consumption",
    "pump_condition",
    "aeration_condition"
]


# ============================================================
# DIRECTORIES
# ============================================================

os.makedirs("outputs/metrics", exist_ok=True)


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("EPBL SMART STP — AI RISK ENGINE")
print("=" * 70)

df = pd.read_csv(DATA_PATH)

df["timestamp"] = pd.to_datetime(df["timestamp"])

df = df.sort_values("timestamp").reset_index(drop=True)

print(f"Dataset loaded: {df.shape}")


# ============================================================
# LOAD SCALER
# ============================================================

scaler = joblib.load(SCALER_PATH)

scaled_data = scaler.transform(
    df[FEATURES].values
)

print("Scaler loaded.")


# ============================================================
# LOAD MODELS
# ============================================================

print("\nLoading LSTM forecasting model...")

lstm_model = load_model(
    LSTM_MODEL_PATH
)

print("LSTM loaded.")


print("\nLoading LSTM autoencoder...")

autoencoder = load_model(
    AUTOENCODER_MODEL_PATH
)

print("Autoencoder loaded.")


# ============================================================
# LOAD THRESHOLD
# ============================================================

with open(THRESHOLD_PATH, "r") as f:

    anomaly_threshold = float(
        f.read().strip()
    )

print(
    f"\nAnomaly threshold: {anomaly_threshold:.6f}"
)


# ============================================================
# CREATE SEQUENCES
# ============================================================

print("\nCreating 24-hour sequences...")

X = []
timestamps = []

for i in range(
    SEQUENCE_LENGTH,
    len(df)
):

    X.append(
        scaled_data[
            i - SEQUENCE_LENGTH:i
        ]
    )

    timestamps.append(
        df["timestamp"].iloc[i]
    )


X = np.array(X)

timestamps = np.array(timestamps)

print(
    f"Sequence dataset: {X.shape}"
)


# ============================================================
# LSTM FORECASTING — BATCH PREDICTION
# ============================================================

print("\nRunning LSTM forecasts...")

forecast_scaled = lstm_model.predict(
    X,
    batch_size=BATCH_SIZE,
    verbose=1
)

forecast_scaled = forecast_scaled.flatten()


# ============================================================
# INVERSE SCALE DO
# ============================================================

DO_INDEX = FEATURES.index("DO")

do_mean = scaler.mean_[DO_INDEX]

do_scale = scaler.scale_[DO_INDEX]

forecast_do = (
    forecast_scaled * do_scale
    + do_mean
)


# ============================================================
# AUTOENCODER — BATCH PREDICTION
# ============================================================

print("\nRunning autoencoder anomaly detection...")

reconstructed = autoencoder.predict(
    X,
    batch_size=BATCH_SIZE,
    verbose=1
)


# ============================================================
# RECONSTRUCTION ERROR
# ============================================================

reconstruction_errors = np.mean(
    np.square(
        X - reconstructed
    ),
    axis=(1, 2)
)


# ============================================================
# ANOMALY DETECTION
# ============================================================

anomaly_detected = (
    reconstruction_errors
    > anomaly_threshold
).astype(int)


# ============================================================
# BUILD RISK RESULTS
# ============================================================

print("\nCalculating risk scores...")

results = []


for idx in range(len(X)):

    original_index = idx + SEQUENCE_LENGTH

    current = df.iloc[original_index]

    current_do = current["DO"]

    current_ph = current["pH"]

    current_flow = current["flow_rate"]

    current_energy = current["energy_consumption"]

    pump_condition = current["pump_condition"]

    aeration_condition = current["aeration_condition"]

    predicted_do = forecast_do[idx]

    reconstruction_error = reconstruction_errors[idx]

    is_anomaly = anomaly_detected[idx]


    # ========================================================
    # RISK SCORE
    # ========================================================

    risk_score = 0

    risk_reasons = []


    # --------------------------------------------------------
    # CURRENT DO
    # --------------------------------------------------------

    if current_do < 1.5:

        risk_score += 3

        risk_reasons.append(
            "Low dissolved oxygen"
        )

    elif current_do < 2.0:

        risk_score += 1

        risk_reasons.append(
            "DO approaching low range"
        )


    # --------------------------------------------------------
    # FORECAST DO
    # --------------------------------------------------------

    if predicted_do < 1.5:

        risk_score += 3

        risk_reasons.append(
            "Forecasted low DO"
        )

    elif predicted_do < 2.0:

        risk_score += 1

        risk_reasons.append(
            "Forecasted DO deterioration"
        )


    # --------------------------------------------------------
    # pH
    # --------------------------------------------------------

    if current_ph < 6.5 or current_ph > 8.5:

        risk_score += 3

        risk_reasons.append(
            "Abnormal pH"
        )


    # --------------------------------------------------------
    # FLOW
    # --------------------------------------------------------

    if current_flow > 1900:

        risk_score += 2

        risk_reasons.append(
            "High flow rate"
        )


    # --------------------------------------------------------
    # ENERGY
    # --------------------------------------------------------

    if current_energy > 90:

        risk_score += 2

        risk_reasons.append(
            "High energy consumption"
        )


    # --------------------------------------------------------
    # PUMP
    # --------------------------------------------------------

    if pump_condition < 0.7:

        risk_score += 3

        risk_reasons.append(
            "Pump condition deterioration"
        )


    # --------------------------------------------------------
    # AERATION
    # --------------------------------------------------------

    if aeration_condition < 0.7:

        risk_score += 3

        risk_reasons.append(
            "Aeration system deterioration"
        )


    # --------------------------------------------------------
    # AI ANOMALY
    # --------------------------------------------------------

    if is_anomaly:

        risk_score += 4

        risk_reasons.append(
            "AI anomaly detected"
        )


    # ========================================================
    # RISK LEVEL
    # ========================================================

    if risk_score >= 7:

        risk_level = "HIGH"

    elif risk_score >= 3:

        risk_level = "MEDIUM"

    else:

        risk_level = "LOW"


    # ========================================================
    # RECOMMENDATION
    # ========================================================

    if risk_level == "HIGH":

        recommendation = (
            "Immediate operator inspection recommended. "
            "Check aeration, pumps and dissolved oxygen."
        )

    elif risk_level == "MEDIUM":

        recommendation = (
            "Monitor STP conditions closely. "
            "Inspect process parameters and equipment."
        )

    else:

        recommendation = (
            "STP operating conditions appear normal. "
            "Continue routine monitoring."
        )


    # ========================================================
    # STORE
    # ========================================================

    results.append({

        "timestamp":
            current["timestamp"],

        "current_DO":
            current_do,

        "forecast_DO":
            predicted_do,

        "pH":
            current_ph,

        "flow_rate":
            current_flow,

        "energy_consumption":
            current_energy,

        "pump_condition":
            pump_condition,

        "aeration_condition":
            aeration_condition,

        "reconstruction_error":
            reconstruction_error,

        "anomaly_detected":
            is_anomaly,

        "risk_score":
            risk_score,

        "risk_level":
            risk_level,

        "risk_reasons":
            "; ".join(risk_reasons),

        "recommendation":
            recommendation
    })


# ============================================================
# SAVE RESULTS
# ============================================================

results_df = pd.DataFrame(results)

results_df.to_csv(
    OUTPUT_PATH,
    index=False
)


# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("RISK ENGINE COMPLETE")
print("=" * 70)


print("\nRisk distribution:")

print(
    results_df[
        "risk_level"
    ].value_counts()
)


print("\nAI anomalies detected:")

print(
    int(
        results_df[
            "anomaly_detected"
        ].sum()
    )
)


print("\nHigh-risk events:")

print(
    int(
        (
            results_df["risk_level"]
            == "HIGH"
        ).sum()
    )
)


print("\nMedium-risk events:")

print(
    int(
        (
            results_df["risk_level"]
            == "MEDIUM"
        ).sum()
    )
)


print("\nLow-risk events:")

print(
    int(
        (
            results_df["risk_level"]
            == "LOW"
        ).sum()
    )
)


print("\nResults saved to:")

print(OUTPUT_PATH)


# ============================================================
# SHOW HIGH-RISK EVENTS
# ============================================================

high_risk = results_df[
    results_df["risk_level"] == "HIGH"
]

if len(high_risk) > 0:

    print("\nTop HIGH-RISK events:")

    print(
        high_risk[
            [
                "timestamp",
                "current_DO",
                "forecast_DO",
                "reconstruction_error",
                "risk_score",
                "risk_reasons"
            ]
        ].head(10).to_string(
            index=False
        )
    )


print("\n" + "=" * 70)

print(
    "NEXT STEP: BUILD STREAMLIT DASHBOARD"
)

print("=" * 70)