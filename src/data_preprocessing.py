
from pathlib import Path

import pandas as pd

DATA_PATH = Path("data/raw/ai4i2020.csv")


def load_data() -> pd.DataFrame:
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH.resolve()}"
        )

    df = pd.read_csv(DATA_PATH)

    # Normalize column names for easier Python usage.
    df.columns = (
        df.columns
        .str.strip()
        .str.replace(r"\s+", "_", regex=True)
        .str.replace(r"[^A-Za-z0-9_]", "", regex=True)
    )

    return df


if __name__ == "__main__":
    df = load_data()

    print("\nDataset shape:", df.shape)
    print("\nColumns:\n", df.columns.tolist())
    print("\nFirst five rows:\n", df.head())
    print("\nMissing values:\n", df.isnull().sum())
    print("\nDuplicate rows:", df.duplicated().sum())

    print("\nTarget distribution:")
    if "Machine_failure" in df.columns:
        print(df["Machine_failure"].value_counts())
    else:
        print("Check the target column name in the downloaded CSV.")
