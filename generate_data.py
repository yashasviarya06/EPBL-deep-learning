import numpy as np
import pandas as pd
from pathlib import Path

# ============================================================
# EPBL SMART STP
# Synthetic STP Data Generator
# ============================================================

# -----------------------------
# Configuration
# -----------------------------

RANDOM_SEED = 42
DAYS = 365
FREQUENCY = "h"

np.random.seed(RANDOM_SEED)

# Create timestamp index
timestamps = pd.date_range(
    start="2025-01-01 00:00:00",
    periods=DAYS * 24,
    freq=FREQUENCY
)

n = len(timestamps)

# Time-related features
hour = timestamps.hour.to_numpy()
day_of_year = timestamps.dayofyear.to_numpy()
day_of_week = timestamps.dayofweek.to_numpy()


# ============================================================
# 1. FLOW RATE
# ============================================================

# Daily usage pattern
daily_flow_pattern = (
    1
    + 0.20 * np.sin(2 * np.pi * (hour - 7) / 24)
)

# Weekly pattern
weekly_pattern = np.where(
    day_of_week < 5,
    1.0,
    0.90
)

# Random noise
flow_noise = np.random.normal(0, 45, n)

flow_rate = (
    1200
    * daily_flow_pattern
    * weekly_pattern
    + flow_noise
)

flow_rate = np.clip(flow_rate, 600, 2200)


# ============================================================
# 2. TEMPERATURE
# ============================================================

seasonal_temperature = (
    27
    + 5 * np.sin(2 * np.pi * (day_of_year - 80) / 365)
)

daily_temperature = (
    2 * np.sin(2 * np.pi * (hour - 14) / 24)
)

temperature = (
    seasonal_temperature
    + daily_temperature
    + np.random.normal(0, 0.5, n)
)

temperature = np.clip(temperature, 15, 40)


# ============================================================
# 3. INFLUENT LOAD
# ============================================================

# Base organic loading influenced by flow
influent_load = (
    1.0
    + 0.0005 * (flow_rate - 1200)
    + np.random.normal(0, 0.05, n)
)

influent_load = np.clip(influent_load, 0.7, 1.5)


# ============================================================
# 4. COD
# ============================================================

cod = (
    180
    + 55 * (influent_load - 1)
    + 10 * np.sin(2 * np.pi * hour / 24)
    + np.random.normal(0, 8, n)
)

cod = np.clip(cod, 100, 350)


# ============================================================
# 5. BOD
# ============================================================

bod = (
    0.48 * cod
    + np.random.normal(0, 5, n)
)

bod = np.clip(bod, 40, 200)


# ============================================================
# 6. TSS
# ============================================================

tss = (
    110
    + 35 * (influent_load - 1)
    + 0.08 * (flow_rate - 1200)
    + np.random.normal(0, 10, n)
)

tss = np.clip(tss, 50, 300)


# ============================================================
# 7. pH
# ============================================================

ph = (
    7.2
    + 0.12 * np.sin(2 * np.pi * hour / 24)
    - 0.08 * (influent_load - 1)
    + np.random.normal(0, 0.05, n)
)

ph = np.clip(ph, 6.5, 8.5)


# ============================================================
# 8. DISSOLVED OXYGEN (DO)
# ============================================================

# Higher organic loading consumes oxygen
do = (
    4.0
    - 0.9 * (influent_load - 1)
    + 0.25 * np.sin(2 * np.pi * hour / 24)
    + np.random.normal(0, 0.15, n)
)

do = np.clip(do, 0.5, 6.5)


# ============================================================
# 9. TDS
# ============================================================

tds = (
    520
    + 90 * (influent_load - 1)
    + np.random.normal(0, 15, n)
)

tds = np.clip(tds, 300, 900)


# ============================================================
# 10. TURBIDITY
# ============================================================

turbidity = (
    22
    + 12 * (influent_load - 1)
    + 0.02 * (tss - 110)
    + np.random.normal(0, 3, n)
)

turbidity = np.clip(turbidity, 5, 100)


# ============================================================
# 11. ORP
# ============================================================

orp = (
    180
    - 25 * (influent_load - 1)
    + 10 * (do - 4)
    + np.random.normal(0, 8, n)
)

orp = np.clip(orp, 50, 300)


# ============================================================
# 12. ENERGY CONSUMPTION
# ============================================================

energy = (
    45
    + 0.018 * flow_rate
    + 8 * (4 - do)
    + np.random.normal(0, 3, n)
)

energy = np.clip(energy, 40, 100)


# ============================================================
# 13. PUMP CONDITION
# ============================================================

pump_condition = np.ones(n)


# ============================================================
# 14. AERATION CONDITION
# ============================================================

aeration_condition = np.ones(n)


# ============================================================
# 15. ANOMALY LABEL
# ============================================================

anomaly = np.zeros(n)


# ============================================================
# INJECT REALISTIC ANOMALIES
# ============================================================

# ------------------------------------------------------------
# Anomaly Type 1: DO collapse
# ------------------------------------------------------------

start = 24 * 45
end = start + 12

do[start:end] *= np.linspace(1, 0.35, end - start)
energy[start:end] *= np.linspace(1, 1.35, end - start)

aeration_condition[start:end] = 0
anomaly[start:end] = 1


# ------------------------------------------------------------
# Anomaly Type 2: pH spike
# ------------------------------------------------------------

start = 24 * 90
end = start + 8

ph[start:end] += np.linspace(0.2, 1.2, end - start)
anomaly[start:end] = 1


# ------------------------------------------------------------
# Anomaly Type 3: Flow surge
# ------------------------------------------------------------

start = 24 * 135
end = start + 10

flow_rate[start:end] *= np.linspace(1, 1.7, end - start)
tss[start:end] *= np.linspace(1, 1.5, end - start)

anomaly[start:end] = 1


# ------------------------------------------------------------
# Anomaly Type 4: Energy spike / pump issue
# ------------------------------------------------------------

start = 24 * 180
end = start + 16

energy[start:end] *= np.linspace(1, 1.8, end - start)

pump_condition[start:end] = 0
anomaly[start:end] = 1


# ------------------------------------------------------------
# Anomaly Type 5: Gradual deterioration
# ------------------------------------------------------------

start = 24 * 230
end = start + 48

cod[start:end] += np.linspace(
    0,
    70,
    end - start
)

do[start:end] -= np.linspace(
    0,
    1.8,
    end - start
)

anomaly[start:end] = 1


# ------------------------------------------------------------
# Anomaly Type 6: Sensor stuck
# ------------------------------------------------------------

start = 24 * 280
end = start + 18

do[start:end] = do[start]

anomaly[start:end] = 1


# ------------------------------------------------------------
# Anomaly Type 7: Multiple simultaneous abnormalities
# ------------------------------------------------------------

start = 24 * 320
end = start + 12

flow_rate[start:end] *= 1.5
cod[start:end] += 60
do[start:end] -= 1.2
energy[start:end] *= 1.4

anomaly[start:end] = 1


# ============================================================
# CREATE DATAFRAME
# ============================================================

data = pd.DataFrame({
    "timestamp": timestamps,
    "pH": ph,
    "temperature": temperature,
    "DO": do,
    "TDS": tds,
    "turbidity": turbidity,
    "ORP": orp,
    "flow_rate": flow_rate,
    "influent_load": influent_load,
    "COD": cod,
    "BOD": bod,
    "TSS": tss,
    "energy_consumption": energy,
    "pump_condition": pump_condition,
    "aeration_condition": aeration_condition,
    "anomaly": anomaly
})


# ============================================================
# SAVE DATA
# ============================================================

output_directory = Path("data")
output_directory.mkdir(exist_ok=True)

output_file = output_directory / "stp_sensor_data.csv"

data.to_csv(output_file, index=False)


# ============================================================
# SUMMARY
# ============================================================

print("=" * 60)
print("EPBL SMART STP - DATA GENERATION COMPLETE")
print("=" * 60)

print(f"Rows generated: {len(data):,}")
print(f"Columns generated: {len(data.columns)}")
print(f"Start date: {data['timestamp'].min()}")
print(f"End date: {data['timestamp'].max()}")

print(
    f"Anomalous observations: "
    f"{int(data['anomaly'].sum()):,}"
)

print(
    f"Normal observations: "
    f"{int((data['anomaly'] == 0).sum()):,}"
)

print("\nDataset preview:")
print(data.head())

print("\nDataset saved to:")
print(output_file)

print("=" * 60)