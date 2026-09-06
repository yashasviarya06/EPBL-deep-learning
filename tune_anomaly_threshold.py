import numpy as np
import pandas as pd
import joblib

from tensorflow.keras.models import load_model
from sklearn.metrics import precision_score, recall_score, f1_score


# ============================================================
# CONFIG
# ============================================================

DATA_PATH = "data/stp_sensor_data.csv"
SCALER_PATH = "models/stp_scaler.pkl"
MODEL_PATH = "models/lstm_autoencoder.keras"

SEQUENCE_LENGTH = 24

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
# LOAD DATA
# ============================================================

print("=" * 70)
print("LOADING DATA")
print("=" * 70)

df = pd.read_csv(DATA_PATH)

df["timestamp"] = pd.to_datetime(df["timestamp"])

df = df.sort_values("timestamp").reset_index(drop=True)


# ============================================================
# LOAD SCALER
# ============================================================

scaler = joblib.load(SCALER_PATH)

scaled_data = scaler.transform(df[FEATURES])


# ============================================================
# CREATE SEQUENCES
# ============================================================

X = []
labels = []

anomaly_labels = df["anomaly"].values

for i in range(SEQUENCE_LENGTH, len(df)):

    sequence = scaled_data[i - SEQUENCE_LENGTH:i]

    X.append(sequence)

    labels.append(anomaly_labels[i])


X = np.array(X)
labels = np.array(labels)


# ============================================================
# TEST SPLIT
# ============================================================

n = len(X)

train_end = int(n * 0.70)
val_end = int(n * 0.85)

X_val = X[train_end:val_end]
X_test = X[val_end:]

y_test = labels[val_end:]


print(f"Validation sequences: {len(X_val)}")
print(f"Test sequences:       {len(X_test)}")


# ============================================================
# LOAD AUTOENCODER
# ============================================================

print("\nLoading LSTM Autoencoder...")

model = load_model(MODEL_PATH)


# ============================================================
# VALIDATION RECONSTRUCTION ERROR
# ============================================================

print("\nCalculating validation reconstruction errors...")

val_reconstructed = model.predict(
    X_val,
    verbose=0
)

val_errors = np.mean(
    np.square(X_val - val_reconstructed),
    axis=(1, 2)
)


# ============================================================
# TEST RECONSTRUCTION ERROR
# ============================================================

print("Calculating test reconstruction errors...")

test_reconstructed = model.predict(
    X_test,
    verbose=0
)

test_errors = np.mean(
    np.square(X_test - test_reconstructed),
    axis=(1, 2)
)


# ============================================================
# TEST DIFFERENT THRESHOLDS
# ============================================================

percentiles = [
    95,
    97,
    98,
    99,
    99.5
]

results = []


print("\n" + "=" * 70)
print("THRESHOLD TUNING RESULTS")
print("=" * 70)

print(
    f"{'Percentile':<12}"
    f"{'Threshold':<14}"
    f"{'Precision':<12}"
    f"{'Recall':<12}"
    f"{'F1':<12}"
    f"{'False Positives':<16}"
)


for percentile in percentiles:

    threshold = np.percentile(
        val_errors,
        percentile
    )

    predictions = (
        test_errors > threshold
    ).astype(int)

    precision = precision_score(
        y_test,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        predictions,
        zero_division=0
    )

    false_positives = np.sum(
        (y_test == 0) & (predictions == 1)
    )

    print(
        f"{percentile:<12}"
        f"{threshold:<14.6f}"
        f"{precision:<12.4f}"
        f"{recall:<12.4f}"
        f"{f1:<12.4f}"
        f"{false_positives:<16}"
    )

    results.append({
        "percentile": percentile,
        "threshold": threshold,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "false_positives": false_positives
    })


# ============================================================
# SAVE RESULTS
# ============================================================

results_df = pd.DataFrame(results)

results_df.to_csv(
    "outputs/metrics/threshold_tuning.csv",
    index=False
)


# ============================================================
# SELECT BEST THRESHOLD
# ============================================================

best_row = results_df.loc[
    results_df["f1_score"].idxmax()
]

print("\n" + "=" * 70)
print("BEST THRESHOLD")
print("=" * 70)

print(
    f"Percentile: {best_row['percentile']}"
)

print(
    f"Threshold: {best_row['threshold']:.6f}"
)

print(
    f"Precision: {best_row['precision']:.4f}"
)

print(
    f"Recall: {best_row['recall']:.4f}"
)

print(
    f"F1 Score: {best_row['f1_score']:.4f}"
)

print(
    f"False Positives: {int(best_row['false_positives'])}"
)

print("\nResults saved to:")
print("outputs/metrics/threshold_tuning.csv")