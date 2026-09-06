import pandas as pd
import numpy as np

from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib


# ============================================================
# EPBL SMART STP
# Time-Series Preprocessing
# ============================================================

print("=" * 70)
print("EPBL SMART STP - PREPROCESSING")
print("=" * 70)


# ============================================================
# 1. LOAD DATA
# ============================================================

input_file = Path("data/stp_sensor_data.csv")

df = pd.read_csv(input_file)

df["timestamp"] = pd.to_datetime(df["timestamp"])

print(f"\nDataset loaded: {df.shape}")


# ============================================================
# 2. SORT BY TIME
# ============================================================

df = df.sort_values("timestamp").reset_index(drop=True)


# ============================================================
# 3. CHECK MISSING VALUES
# ============================================================

missing_values = df.isnull().sum().sum()

print(f"Missing values: {missing_values}")

if missing_values > 0:
    print("Missing values detected. Filling using forward fill...")
    df = df.ffill().bfill()


# ============================================================
# 4. DEFINE FEATURES
# ============================================================

features = [
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

target = "DO"


print("\nFeatures used:")
for feature in features:
    print(f"- {feature}")

print(f"\nPrediction target: {target}")


# ============================================================
# 5. EXTRACT FEATURE MATRIX
# ============================================================

X = df[features].values

y = df[target].values


# ============================================================
# 6. TEMPORAL TRAIN / TEST SPLIT
# ============================================================

# IMPORTANT:
# For time-series data, we do NOT randomly shuffle the data.

train_size = int(len(df) * 0.70)

validation_size = int(len(df) * 0.15)

test_start = train_size + validation_size

X_train_raw = X[:train_size]

X_val_raw = X[train_size:test_start]

X_test_raw = X[test_start:]


print("\nTemporal split:")
print(f"Training observations:   {len(X_train_raw):,}")
print(f"Validation observations: {len(X_val_raw):,}")
print(f"Testing observations:    {len(X_test_raw):,}")


# ============================================================
# 7. SCALE FEATURES
# ============================================================

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train_raw)

X_val_scaled = scaler.transform(X_val_raw)

X_test_scaled = scaler.transform(X_test_raw)


# ============================================================
# 8. SAVE SCALER
# ============================================================

model_directory = Path("models")

model_directory.mkdir(exist_ok=True)

scaler_file = model_directory / "stp_scaler.pkl"

joblib.dump(scaler, scaler_file)

print(f"\nScaler saved to: {scaler_file}")


# ============================================================
# 9. CREATE TIME-SERIES SEQUENCES
# ============================================================

SEQUENCE_LENGTH = 24

FORECAST_HORIZON = 1


def create_sequences(data, target_data, sequence_length, horizon):

    X_sequences = []
    y_sequences = []

    for i in range(
        sequence_length,
        len(data) - horizon + 1
    ):

        X_sequences.append(
            data[i - sequence_length:i]
        )

        y_sequences.append(
            target_data[i + horizon - 1]
        )

    return (
        np.array(X_sequences),
        np.array(y_sequences)
    )


# Target needs to be scaled using the DO column scaler.

do_index = features.index("DO")

y_train_scaled = X_train_scaled[:, do_index]

y_val_scaled = X_val_scaled[:, do_index]

y_test_scaled = X_test_scaled[:, do_index]


X_train, y_train = create_sequences(
    X_train_scaled,
    y_train_scaled,
    SEQUENCE_LENGTH,
    FORECAST_HORIZON
)

X_val, y_val = create_sequences(
    X_val_scaled,
    y_val_scaled,
    SEQUENCE_LENGTH,
    FORECAST_HORIZON
)

X_test, y_test = create_sequences(
    X_test_scaled,
    y_test_scaled,
    SEQUENCE_LENGTH,
    FORECAST_HORIZON
)


# ============================================================
# 10. DISPLAY SHAPES
# ============================================================

print("\n" + "=" * 70)
print("SEQUENCE SHAPES")
print("=" * 70)

print(f"X_train: {X_train.shape}")
print(f"y_train: {y_train.shape}")

print(f"X_val:   {X_val.shape}")
print(f"y_val:   {y_val.shape}")

print(f"X_test:  {X_test.shape}")
print(f"y_test:  {y_test.shape}")


# ============================================================
# 11. SAVE PROCESSED DATA
# ============================================================

processed_directory = Path("data/processed")

processed_directory.mkdir(
    parents=True,
    exist_ok=True
)


np.save(
    processed_directory / "X_train.npy",
    X_train
)

np.save(
    processed_directory / "y_train.npy",
    y_train
)

np.save(
    processed_directory / "X_val.npy",
    X_val
)

np.save(
    processed_directory / "y_val.npy",
    y_val
)

np.save(
    processed_directory / "X_test.npy",
    X_test
)

np.save(
    processed_directory / "y_test.npy",
    y_test
)


# ============================================================
# 12. SAVE FEATURE LIST
# ============================================================

with open(
    processed_directory / "features.txt",
    "w"
) as f:

    for feature in features:
        f.write(feature + "\n")


# ============================================================
# 13. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("PREPROCESSING COMPLETE")
print("=" * 70)

print(f"Sequence length: {SEQUENCE_LENGTH} hours")
print(f"Forecast horizon: {FORECAST_HORIZON} hour")

print("\nModel input:")
print(
    f"{SEQUENCE_LENGTH} hours × "
    f"{len(features)} features"
)

print("\nTarget:")
print("Next-hour dissolved oxygen (DO)")

print("\nProcessed files saved to:")
print(processed_directory)

print("=" * 70)