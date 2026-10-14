
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

DATA_PATH = Path("data/raw/ai4i2020.csv")
REPORT_DIR = Path("reports")
FIGURE_DIR = REPORT_DIR / "figures"

FIGURE_DIR.mkdir(parents=True, exist_ok=True)
sns.set_theme(style="whitegrid")


def load_data():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH.resolve()}"
        )

    df = pd.read_csv(DATA_PATH)

    df.columns = (
        df.columns.str.strip()
        .str.replace(r"\s+", "_", regex=True)
        .str.replace(r"[^A-Za-z0-9_]", "", regex=True)
    )

    return df


def run_eda():
    df = load_data()

    target = "Machine_failure"

    if target not in df.columns:
        raise ValueError(
            f"Target '{target}' not found. "
            f"Available columns: {df.columns.tolist()}"
        )

    # 1. Dataset overview
    print("\n--- DATASET OVERVIEW ---")
    print("Shape:", df.shape)
    print("\nData types:\n", df.dtypes)
    print("\nMissing values:\n", df.isnull().sum())
    print("\nDuplicate rows:", df.duplicated().sum())

    # 2. Target distribution
    print("\n--- FAILURE DISTRIBUTION ---")
    print(df[target].value_counts(dropna=False))
    print("\nFailure rate (%):")
    print(df[target].value_counts(normalize=True) * 100)

    plt.figure(figsize=(7, 5))
    sns.countplot(data=df, x=target)
    plt.title("Machine Failure Distribution")
    plt.xlabel("Machine Failure (0 = No, 1 = Yes)")
    plt.ylabel("Number of Records")
    plt.tight_layout()
    plt.savefig(
        FIGURE_DIR / "failure_distribution.png",
        dpi=150
    )
    plt.close()

    # 3. Numeric feature distributions
    numeric_cols = df.select_dtypes(
        include="number"
    ).columns.tolist()

    exclude = [
        "UDI",
        "Product_ID",
        "Machine_failure",
        "TWF",
        "HDF",
        "PWF",
        "OSF",
        "RNF",
    ]

    sensor_cols = [
        col for col in numeric_cols
        if col not in exclude
    ]

    for col in sensor_cols:
        plt.figure(figsize=(7, 4))
        sns.histplot(data=df, x=col, kde=True)
        plt.title(f"Distribution of {col}")
        plt.tight_layout()
        filename = col.lower() + "_distribution.png"
        plt.savefig(FIGURE_DIR / filename, dpi=150)
        plt.close()

    # 4. Sensor distributions by target
    for col in sensor_cols:
        plt.figure(figsize=(7, 4))
        sns.boxplot(data=df, x=target, y=col)
        plt.title(f"{col} by Machine Failure")
        plt.tight_layout()
        filename = col.lower() + "_by_failure.png"
        plt.savefig(FIGURE_DIR / filename, dpi=150)
        plt.close()

    # 5. Correlation heatmap
    correlation_cols = sensor_cols + [target]
    corr = df[correlation_cols].corr()

    plt.figure(figsize=(11, 8))
    sns.heatmap(
        corr,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0
    )
    plt.title("Sensor and Machine Failure Correlations")
    plt.tight_layout()
    plt.savefig(
        FIGURE_DIR / "correlation_heatmap.png",
        dpi=150
    )
    plt.close()

    # 6. Save quality report
    quality = pd.DataFrame({
        "dtype": df.dtypes.astype(str),
        "missing_values": df.isnull().sum(),
        "unique_values": df.nunique(),
    })

    quality.to_csv(REPORT_DIR / "data_quality_report.csv")

    print(f"\nEDA complete. Charts saved to: {FIGURE_DIR}")
    print(f"Quality report saved to: {REPORT_DIR}")


if __name__ == "__main__":
    run_eda()
