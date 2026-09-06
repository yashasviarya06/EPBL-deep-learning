import pandas as pd
import numpy as np

from pathlib import Path

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

import joblib


# ============================================================
# EPBL SMART STP
# Baseline Machine Learning Model
# ============================================================

print("=" * 70)
print("EPBL SMART STP - BASELINE RANDOM FOREST")
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


print("\nOriginal sequence shapes:")

print(f"X_train: {X_train.shape}")
print(f"X_val:   {X_val.shape}")
print(f"X_test:  {X_test.shape}")


# ============================================================
# 2. CONVERT TIME-SERIES SEQUENCES TO TABULAR DATA
# ============================================================

# Random Forest does not understand 3D sequences.
#
# We therefore use the most recent observation from
# each 24-hour sequence as the baseline representation.

X_train_baseline = X_train[:, -1, :]
X_val_baseline = X_val[:, -1, :]
X_test_baseline = X_test[:, -1, :]


print("\nBaseline input shapes:")

print(f"X_train: {X_train_baseline.shape}")
print(f"X_val:   {X_val_baseline.shape}")
print(f"X_test:  {X_test_baseline.shape}")


# ============================================================
# 3. TRAIN RANDOM FOREST
# ============================================================

print("\nTraining Random Forest...")

model = RandomForestRegressor(
    n_estimators=200,
    max_depth=12,
    min_samples_split=5,
    random_state=42,
    n_jobs=-1
)

model.fit(
    X_train_baseline,
    y_train
)

print("Training complete.")


# ============================================================
# 4. VALIDATION PREDICTION
# ============================================================

y_val_pred = model.predict(X_val_baseline)

val_mae = mean_absolute_error(
    y_val,
    y_val_pred
)

val_rmse = np.sqrt(
    mean_squared_error(
        y_val,
        y_val_pred
    )
)

val_r2 = r2_score(
    y_val,
    y_val_pred
)


print("\nValidation performance:")

print(f"MAE:  {val_mae:.4f}")
print(f"RMSE: {val_rmse:.4f}")
print(f"R²:   {val_r2:.4f}")


# ============================================================
# 5. TEST PREDICTION
# ============================================================

y_test_pred = model.predict(X_test_baseline)


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
print("TEST PERFORMANCE")
print("=" * 70)

print(f"MAE:  {test_mae:.4f}")
print(f"RMSE: {test_rmse:.4f}")
print(f"R²:   {test_r2:.4f}")


# ============================================================
# 6. SAVE MODEL
# ============================================================

model_dir = Path("models")

model_dir.mkdir(
    exist_ok=True
)

model_file = model_dir / "random_forest_baseline.pkl"

joblib.dump(
    model,
    model_file
)

print(f"\nModel saved to:")
print(model_file)


# ============================================================
# 7. SAVE METRICS
# ============================================================

metrics_dir = Path("outputs/metrics")

metrics_dir.mkdir(
    parents=True,
    exist_ok=True
)

metrics = pd.DataFrame({
    "model": ["Random Forest"],
    "MAE": [test_mae],
    "RMSE": [test_rmse],
    "R2": [test_r2]
})

metrics_file = metrics_dir / "baseline_metrics.csv"

metrics.to_csv(
    metrics_file,
    index=False
)

print(f"Metrics saved to:")
print(metrics_file)


# ============================================================
# 8. FEATURE IMPORTANCE
# ============================================================

features_file = processed_dir / "features.txt"

with open(features_file, "r") as f:
    features = [
        line.strip()
        for line in f.readlines()
    ]


importance = pd.DataFrame({
    "feature": features,
    "importance": model.feature_importances_
})

importance = importance.sort_values(
    "importance",
    ascending=False
)


print("\n" + "=" * 70)
print("FEATURE IMPORTANCE")
print("=" * 70)

print(
    importance.to_string(
        index=False
    )
)


importance.to_csv(
    metrics_dir / "baseline_feature_importance.csv",
    index=False
)


# ============================================================
# FINAL
# ============================================================

print("\n" + "=" * 70)
print("BASELINE MODEL COMPLETE")
print("=" * 70)

print("\nWe now have a traditional ML benchmark")
print("against which the LSTM can be compared.")

print("=" * 70)