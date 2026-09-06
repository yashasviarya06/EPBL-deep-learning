import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


# ============================================================
# EPBL SMART STP
# LSTM FORECASTING MODEL
# ============================================================

print("=" * 70)
print("EPBL SMART STP - LSTM FORECASTING")
print("=" * 70)


# ============================================================
# 1. LOAD PROCESSED DATA
# ============================================================

processed_dir = Path("data/processed")

X_train = np.load(processed_dir / "X_train.npy")
y_train = np.load(processed_dir / "y_train.npy")

X_val = np.load(processed_dir / "X_val.npy")
y_val = np.load(processed_dir / "y_val.npy")

X_test = np.load(processed_dir / "X_test.npy")
y_test = np.load(processed_dir / "y_test.npy")


print("\nDataset shapes:")

print(f"X_train: {X_train.shape}")
print(f"y_train: {y_train.shape}")

print(f"X_val:   {X_val.shape}")
print(f"y_val:   {y_val.shape}")

print(f"X_test:  {X_test.shape}")
print(f"y_test:  {y_test.shape}")


# ============================================================
# 2. BUILD LSTM MODEL
# ============================================================

sequence_length = X_train.shape[1]

number_of_features = X_train.shape[2]


model = Sequential([

    LSTM(
        64,
        return_sequences=True,
        input_shape=(
            sequence_length,
            number_of_features
        )
    ),

    Dropout(0.2),

    LSTM(
        32,
        return_sequences=False
    ),

    Dropout(0.2),

    Dense(16, activation="relu"),

    Dense(1)

])


# ============================================================
# 3. COMPILE MODEL
# ============================================================

model.compile(
    optimizer="adam",
    loss="mse",
    metrics=["mae"]
)


print("\n" + "=" * 70)
print("MODEL ARCHITECTURE")
print("=" * 70)

model.summary()


# ============================================================
# 4. CALLBACKS
# ============================================================

early_stopping = EarlyStopping(
    monitor="val_loss",
    patience=8,
    restore_best_weights=True
)


reduce_learning_rate = ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.5,
    patience=4,
    min_lr=0.00001
)


# ============================================================
# 5. TRAIN MODEL
# ============================================================

print("\n" + "=" * 70)
print("TRAINING LSTM")
print("=" * 70)

history = model.fit(

    X_train,
    y_train,

    validation_data=(
        X_val,
        y_val
    ),

    epochs=50,

    batch_size=64,

    callbacks=[
        early_stopping,
        reduce_learning_rate
    ],

    verbose=1

)


# ============================================================
# 6. TEST PREDICTIONS
# ============================================================

print("\nGenerating test predictions...")

y_test_pred = model.predict(
    X_test,
    verbose=0
).flatten()


# ============================================================
# 7. EVALUATE MODEL
# ============================================================

test_mae = mean_absolute_error(
    y_test,
    y_test_pred
)

test_rmse = np.sqrt(
    mean_squared_error(
        y_test,
        y_test_pred
    )
)

test_r2 = r2_score(
    y_test,
    y_test_pred
)


print("\n" + "=" * 70)
print("LSTM TEST PERFORMANCE")
print("=" * 70)

print(f"MAE:  {test_mae:.4f}")
print(f"RMSE: {test_rmse:.4f}")
print(f"R²:   {test_r2:.4f}")


# ============================================================
# 8. SAVE MODEL
# ============================================================

model_dir = Path("models")

model_dir.mkdir(
    exist_ok=True
)

model_file = model_dir / "lstm_stp_forecaster.keras"

model.save(model_file)

print(f"\nLSTM model saved to:")
print(model_file)


# ============================================================
# 9. SAVE METRICS
# ============================================================

metrics_dir = Path("outputs/metrics")

metrics_dir.mkdir(
    parents=True,
    exist_ok=True
)

metrics = pd.DataFrame({

    "model": [
        "Random Forest",
        "LSTM"
    ],

    "MAE": [
        0.4883,
        test_mae
    ],

    "RMSE": [
        0.6272,
        test_rmse
    ],

    "R2": [
        0.5966,
        test_r2
    ]

})

metrics_file = metrics_dir / "model_comparison.csv"

metrics.to_csv(
    metrics_file,
    index=False
)

print(f"\nComparison saved to:")
print(metrics_file)


# ============================================================
# 10. TRAINING LOSS CURVE
# ============================================================

plot_dir = Path("outputs/plots")

plot_dir.mkdir(
    parents=True,
    exist_ok=True
)


plt.figure(figsize=(10, 5))

plt.plot(
    history.history["loss"],
    label="Training Loss"
)

plt.plot(
    history.history["val_loss"],
    label="Validation Loss"
)

plt.title("LSTM Training and Validation Loss")

plt.xlabel("Epoch")

plt.ylabel("MSE Loss")

plt.legend()

plt.grid(alpha=0.3)

plt.tight_layout()

plt.savefig(
    plot_dir / "lstm_training_loss.png",
    dpi=150
)

plt.close()


# ============================================================
# 11. ACTUAL VS PREDICTED
# ============================================================

plt.figure(figsize=(14, 5))

sample_size = min(
    500,
    len(y_test)
)

plt.plot(
    y_test[:sample_size],
    label="Actual DO"
)

plt.plot(
    y_test_pred[:sample_size],
    label="Predicted DO"
)

plt.title(
    "LSTM: Actual vs Predicted Dissolved Oxygen"
)

plt.xlabel("Test Time Step")

plt.ylabel("Scaled DO")

plt.legend()

plt.grid(alpha=0.3)

plt.tight_layout()

plt.savefig(
    plot_dir / "lstm_actual_vs_predicted.png",
    dpi=150
)

plt.close()


# ============================================================
# 12. FINAL
# ============================================================

print("\n" + "=" * 70)
print("LSTM TRAINING COMPLETE")
print("=" * 70)

print("\nGenerated files:")

print("- models/lstm_stp_forecaster.keras")
print("- outputs/metrics/model_comparison.csv")
print("- outputs/plots/lstm_training_loss.png")
print("- outputs/plots/lstm_actual_vs_predicted.png")

print("\nNext step:")
print("Build the LSTM Autoencoder for anomaly detection.")

print("=" * 70)