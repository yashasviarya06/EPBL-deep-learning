import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    LSTM,
    RepeatVector,
    TimeDistributed,
    Dense,
    Dropout
)
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    mean_squared_error
)


# ============================================================
# 1. CONFIGURATION
# ============================================================

DATA_PATH = "data/stp_sensor_data.csv"
SCALER_PATH = "models/stp_scaler.pkl"

MODEL_PATH = "models/lstm_autoencoder.keras"
METRICS_PATH = "outputs/metrics/anomaly_metrics.csv"
THRESHOLD_PATH = "outputs/metrics/anomaly_threshold.txt"

PLOT_ERROR = "outputs/plots/anomaly_reconstruction_error.png"
PLOT_CONFUSION = "outputs/plots/anomaly_confusion_matrix.png"

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
# 2. CREATE OUTPUT DIRECTORIES
# ============================================================

os.makedirs("models", exist_ok=True)
os.makedirs("outputs/metrics", exist_ok=True)
os.makedirs("outputs/plots", exist_ok=True)


# ============================================================
# 3. LOAD DATA
# ============================================================

print("=" * 70)
print("LOADING STP DATA")
print("=" * 70)

df = pd.read_csv(DATA_PATH)

df["timestamp"] = pd.to_datetime(df["timestamp"])

df = df.sort_values("timestamp").reset_index(drop=True)

print(f"Dataset shape: {df.shape}")


# ============================================================
# 4. LOAD TRAINING SCALER
# ============================================================

scaler = joblib.load(SCALER_PATH)

scaled_features = scaler.transform(df[FEATURES])

print("Scaler loaded successfully.")


# ============================================================
# 5. CREATE SEQUENCES
# ============================================================

X = []
labels = []
timestamps = []

anomaly_labels = df["anomaly"].values

for i in range(SEQUENCE_LENGTH, len(df)):

    sequence = scaled_features[i - SEQUENCE_LENGTH:i]

    X.append(sequence)

    # Label the sequence using the current hour.
    labels.append(anomaly_labels[i])

    timestamps.append(df["timestamp"].iloc[i])


X = np.array(X)
labels = np.array(labels)
timestamps = np.array(timestamps)

print(f"Sequence shape: {X.shape}")


# ============================================================
# 6. TEMPORAL TRAIN / VALIDATION / TEST SPLIT
# ============================================================

n = len(X)

train_end = int(n * 0.70)
val_end = int(n * 0.85)

X_train_all = X[:train_end]
y_train_labels = labels[:train_end]

X_val = X[train_end:val_end]
y_val_labels = labels[train_end:val_end]

X_test = X[val_end:]
y_test_labels = labels[val_end:]

print("\nDataset split:")
print(f"Training sequences:   {len(X_train_all)}")
print(f"Validation sequences: {len(X_val)}")
print(f"Test sequences:       {len(X_test)}")


# ============================================================
# 7. KEEP ONLY NORMAL TRAINING SEQUENCES
# ============================================================

# To prevent the autoencoder from learning abnormal behaviour,
# keep sequences where ALL 24 hours are normal.

normal_train_indices = []

for i in range(SEQUENCE_LENGTH, train_end + SEQUENCE_LENGTH):

    window_labels = anomaly_labels[i - SEQUENCE_LENGTH:i]

    if np.max(window_labels) == 0:
        normal_train_indices.append(i - SEQUENCE_LENGTH)


X_train = X_train_all[normal_train_indices]

print("\nNormal training sequences:")
print(f"{len(X_train)}")


# ============================================================
# 8. BUILD LSTM AUTOENCODER
# ============================================================

print("\n" + "=" * 70)
print("BUILDING LSTM AUTOENCODER")
print("=" * 70)

model = Sequential([

    LSTM(
        64,
        activation="tanh",
        return_sequences=True,
        input_shape=(SEQUENCE_LENGTH, len(FEATURES))
    ),

    Dropout(0.2),

    LSTM(
        32,
        activation="tanh",
        return_sequences=False
    ),

    RepeatVector(SEQUENCE_LENGTH),

    LSTM(
        32,
        activation="tanh",
        return_sequences=True
    ),

    Dropout(0.2),

    LSTM(
        64,
        activation="tanh",
        return_sequences=True
    ),

    TimeDistributed(
        Dense(len(FEATURES))
    )
])


model.compile(
    optimizer="adam",
    loss="mse"
)

model.summary()


# ============================================================
# 9. CALLBACKS
# ============================================================

early_stopping = EarlyStopping(
    monitor="val_loss",
    patience=8,
    restore_best_weights=True
)

reduce_lr = ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.5,
    patience=4,
    min_lr=0.00001
)


# ============================================================
# 10. TRAIN AUTOENCODER
# ============================================================

print("\n" + "=" * 70)
print("TRAINING AUTOENCODER")
print("=" * 70)

history = model.fit(
    X_train,
    X_train,
    validation_data=(X_val, X_val),
    epochs=50,
    batch_size=64,
    callbacks=[
        early_stopping,
        reduce_lr
    ],
    verbose=1
)


# ============================================================
# 11. SAVE MODEL
# ============================================================

model.save(MODEL_PATH)

print("\nAutoencoder saved to:")
print(MODEL_PATH)


# ============================================================
# 12. CALCULATE VALIDATION RECONSTRUCTION ERROR
# ============================================================

print("\n" + "=" * 70)
print("CALCULATING ANOMALY THRESHOLD")
print("=" * 70)

val_predictions = model.predict(
    X_val,
    verbose=0
)

val_errors = np.mean(
    np.square(X_val - val_predictions),
    axis=(1, 2)
)


# ============================================================
# 13. SELECT THRESHOLD
# ============================================================

# 95th percentile of validation reconstruction error.

threshold = np.percentile(
    val_errors,
    95
)

print(f"Anomaly threshold: {threshold:.6f}")

with open(THRESHOLD_PATH, "w") as f:
    f.write(str(threshold))


# ============================================================
# 14. TEST RECONSTRUCTION
# ============================================================

print("\n" + "=" * 70)
print("TESTING ANOMALY DETECTION")
print("=" * 70)

test_predictions = model.predict(
    X_test,
    verbose=0
)

test_errors = np.mean(
    np.square(X_test - test_predictions),
    axis=(1, 2)
)


# ============================================================
# 15. GENERATE ANOMALY PREDICTIONS
# ============================================================

y_pred = (
    test_errors > threshold
).astype(int)


# ============================================================
# 16. EVALUATION
# ============================================================

precision = precision_score(
    y_test_labels,
    y_pred,
    zero_division=0
)

recall = recall_score(
    y_test_labels,
    y_pred,
    zero_division=0
)

f1 = f1_score(
    y_test_labels,
    y_pred,
    zero_division=0
)

cm = confusion_matrix(
    y_test_labels,
    y_pred
)


print("\nANOMALY DETECTION PERFORMANCE")
print("-" * 50)

print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1 Score:  {f1:.4f}")

print("\nConfusion Matrix:")
print(cm)


# ============================================================
# 17. FALSE POSITIVE / FALSE NEGATIVE
# ============================================================

tn, fp, fn, tp = cm.ravel()

false_positive_rate = fp / (fp + tn) if (fp + tn) > 0 else 0
false_negative_rate = fn / (fn + tp) if (fn + tp) > 0 else 0

print(f"\nFalse Positive Rate: {false_positive_rate:.4f}")
print(f"False Negative Rate: {false_negative_rate:.4f}")


# ============================================================
# 18. SAVE METRICS
# ============================================================

metrics = pd.DataFrame({
    "metric": [
        "precision",
        "recall",
        "f1_score",
        "false_positive_rate",
        "false_negative_rate",
        "threshold"
    ],

    "value": [
        precision,
        recall,
        f1,
        false_positive_rate,
        false_negative_rate,
        threshold
    ]
})

metrics.to_csv(
    METRICS_PATH,
    index=False
)


# ============================================================
# 19. PLOT RECONSTRUCTION ERROR
# ============================================================

plt.figure(figsize=(14, 6))

plt.plot(
    timestamps[val_end:],
    test_errors,
    linewidth=1
)

plt.axhline(
    threshold,
    linestyle="--",
    label=f"Threshold = {threshold:.4f}"
)

anomaly_times = timestamps[val_end:][y_test_labels == 1]

for t in anomaly_times:
    plt.axvline(
        t,
        alpha=0.15
    )

plt.title(
    "LSTM Autoencoder Reconstruction Error"
)

plt.xlabel(
    "Time"
)

plt.ylabel(
    "Reconstruction Error"
)

plt.legend()

plt.xticks(rotation=45)

plt.tight_layout()

plt.savefig(
    PLOT_ERROR,
    dpi=150
)

plt.close()


# ============================================================
# 20. CONFUSION MATRIX PLOT
# ============================================================

plt.figure(figsize=(6, 5))

plt.imshow(
    cm,
    interpolation="nearest"
)

plt.title(
    "Anomaly Detection Confusion Matrix"
)

plt.colorbar()

plt.xticks(
    [0, 1],
    ["Normal", "Anomaly"]
)

plt.yticks(
    [0, 1],
    ["Normal", "Anomaly"]
)

plt.xlabel(
    "Predicted"
)

plt.ylabel(
    "Actual"
)

for i in range(2):
    for j in range(2):

        plt.text(
            j,
            i,
            cm[i, j],
            ha="center",
            va="center"
        )

plt.tight_layout()

plt.savefig(
    PLOT_CONFUSION,
    dpi=150
)

plt.close()


# ============================================================
# 21. FINAL OUTPUT
# ============================================================

print("\n" + "=" * 70)
print("AUTOENCODER TRAINING COMPLETE")
print("=" * 70)

print("\nGenerated files:")

print(f"- {MODEL_PATH}")
print(f"- {THRESHOLD_PATH}")
print(f"- {METRICS_PATH}")
print(f"- {PLOT_ERROR}")
print(f"- {PLOT_CONFUSION}")

print("\nNext step:")
print("Build the AI Risk Engine.")