import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ============================================================
# EPBL SMART STP
# Exploratory Data Analysis
# ============================================================

print("=" * 70)
print("EPBL SMART STP - EXPLORATORY DATA ANALYSIS")
print("=" * 70)

# ------------------------------------------------------------
# 1. Load dataset
# ------------------------------------------------------------

file_path = Path("data/stp_sensor_data.csv")

df = pd.read_csv(file_path)

df["timestamp"] = pd.to_datetime(df["timestamp"])

print("\nDataset loaded successfully.")

# ------------------------------------------------------------
# 2. Basic information
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("DATASET INFORMATION")
print("=" * 70)

print(f"Rows: {len(df):,}")
print(f"Columns: {len(df.columns)}")

print("\nColumns:")
for column in df.columns:
    print(f"- {column}")

print("\nData types:")
print(df.dtypes)

# ------------------------------------------------------------
# 3. Missing values
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("MISSING VALUES")
print("=" * 70)

missing = df.isnull().sum()

print(missing)

print(f"\nTotal missing values: {missing.sum()}")

# ------------------------------------------------------------
# 4. Duplicate rows
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("DUPLICATES")
print("=" * 70)

duplicates = df.duplicated().sum()

print(f"Duplicate rows: {duplicates}")

# ------------------------------------------------------------
# 5. Statistical summary
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("STATISTICAL SUMMARY")
print("=" * 70)

print(df.describe().round(2))

# ------------------------------------------------------------
# 6. Anomaly distribution
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("ANOMALY DISTRIBUTION")
print("=" * 70)

normal_count = (df["anomaly"] == 0).sum()
anomaly_count = (df["anomaly"] == 1).sum()

print(f"Normal observations:    {normal_count:,}")
print(f"Anomalous observations: {anomaly_count:,}")

anomaly_percentage = anomaly_count / len(df) * 100

print(f"Anomaly percentage:     {anomaly_percentage:.2f}%")

# ------------------------------------------------------------
# 7. Create output directory
# ------------------------------------------------------------

plot_directory = Path("outputs/plots")
plot_directory.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------
# 8. Plot pH
# ------------------------------------------------------------

plt.figure(figsize=(14, 5))

plt.plot(
    df["timestamp"],
    df["pH"],
    linewidth=0.8
)

plt.title("STP pH Over Time")
plt.xlabel("Time")
plt.ylabel("pH")
plt.grid(alpha=0.3)

plt.tight_layout()

plt.savefig(
    plot_directory / "ph_over_time.png",
    dpi=150
)

plt.close()

# ------------------------------------------------------------
# 9. Plot dissolved oxygen
# ------------------------------------------------------------

plt.figure(figsize=(14, 5))

plt.plot(
    df["timestamp"],
    df["DO"],
    linewidth=0.8
)

plt.title("Dissolved Oxygen (DO) Over Time")
plt.xlabel("Time")
plt.ylabel("DO (mg/L)")
plt.grid(alpha=0.3)

plt.tight_layout()

plt.savefig(
    plot_directory / "do_over_time.png",
    dpi=150
)

plt.close()

# ------------------------------------------------------------
# 10. Plot COD
# ------------------------------------------------------------

plt.figure(figsize=(14, 5))

plt.plot(
    df["timestamp"],
    df["COD"],
    linewidth=0.8
)

plt.title("COD Over Time")
plt.xlabel("Time")
plt.ylabel("COD")
plt.grid(alpha=0.3)

plt.tight_layout()

plt.savefig(
    plot_directory / "cod_over_time.png",
    dpi=150
)

plt.close()

# ------------------------------------------------------------
# 11. Plot flow rate
# ------------------------------------------------------------

plt.figure(figsize=(14, 5))

plt.plot(
    df["timestamp"],
    df["flow_rate"],
    linewidth=0.8
)

plt.title("STP Flow Rate Over Time")
plt.xlabel("Time")
plt.ylabel("Flow Rate")
plt.grid(alpha=0.3)

plt.tight_layout()

plt.savefig(
    plot_directory / "flow_rate_over_time.png",
    dpi=150
)

plt.close()

# ------------------------------------------------------------
# 12. Plot energy consumption
# ------------------------------------------------------------

plt.figure(figsize=(14, 5))

plt.plot(
    df["timestamp"],
    df["energy_consumption"],
    linewidth=0.8
)

plt.title("Energy Consumption Over Time")
plt.xlabel("Time")
plt.ylabel("Energy Consumption")
plt.grid(alpha=0.3)

plt.tight_layout()

plt.savefig(
    plot_directory / "energy_over_time.png",
    dpi=150
)

plt.close()

# ------------------------------------------------------------
# 13. Correlation matrix
# ------------------------------------------------------------

numeric_df = df.drop(columns=["timestamp"])

correlation = numeric_df.corr()

plt.figure(figsize=(13, 10))

plt.imshow(
    correlation,
    aspect="auto"
)

plt.colorbar(label="Correlation")

plt.xticks(
    range(len(correlation.columns)),
    correlation.columns,
    rotation=90
)

plt.yticks(
    range(len(correlation.columns)),
    correlation.columns
)

plt.title("STP Sensor Correlation Matrix")

plt.tight_layout()

plt.savefig(
    plot_directory / "correlation_matrix.png",
    dpi=150
)

plt.close()

# ------------------------------------------------------------
# 14. Distribution of anomalies
# ------------------------------------------------------------

plt.figure(figsize=(7, 5))

plt.bar(
    ["Normal", "Anomaly"],
    [normal_count, anomaly_count]
)

plt.title("Normal vs Anomalous Observations")
plt.ylabel("Number of Observations")

plt.tight_layout()

plt.savefig(
    plot_directory / "anomaly_distribution.png",
    dpi=150
)

plt.close()

# ------------------------------------------------------------
# 15. Anomaly periods
# ------------------------------------------------------------

anomaly_df = df[df["anomaly"] == 1]

plt.figure(figsize=(14, 5))

plt.plot(
    df["timestamp"],
    df["DO"],
    linewidth=0.8,
    label="DO"
)

plt.scatter(
    anomaly_df["timestamp"],
    anomaly_df["DO"],
    s=8,
    label="Anomaly"
)

plt.title("DO and Detected Anomaly Periods")
plt.xlabel("Time")
plt.ylabel("DO (mg/L)")
plt.legend()
plt.grid(alpha=0.3)

plt.tight_layout()

plt.savefig(
    plot_directory / "do_anomaly_periods.png",
    dpi=150
)

plt.close()

# ------------------------------------------------------------
# 16. Check realistic relationships
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("KEY CORRELATIONS")
print("=" * 70)

relationships = [
    ("influent_load", "COD"),
    ("influent_load", "DO"),
    ("flow_rate", "TSS"),
    ("DO", "energy_consumption"),
    ("COD", "BOD")
]

for x, y in relationships:

    correlation_value = df[x].corr(df[y])

    print(
        f"{x:20s} vs {y:20s}: "
        f"{correlation_value:.3f}"
    )

# ------------------------------------------------------------
# 17. Final report
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("EDA COMPLETE")
print("=" * 70)

print("\nPlots saved to:")
print(plot_directory)

print("\nGenerated files:")

for file in sorted(plot_directory.glob("*.png")):
    print(f"- {file.name}")

print("\nDataset is ready for preprocessing and model development.")
print("=" * 70)