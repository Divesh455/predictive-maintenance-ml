
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

DATA_PATH = Path("data/raw/ai4i2020.csv")
OUTPUT_DIR = Path("data/processed")

TARGET = "Machine_failure"
RANDOM_STATE = 42
TEST_SIZE = 0.20


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH.resolve()}"
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(DATA_PATH)

    # Normalize column names to match our EDA script.
    df.columns = (
        df.columns.str.strip()
        .str.replace(r"\s+", "_", regex=True)
        .str.replace(r"[^A-Za-z0-9_]", "", regex=True)
    )

    if TARGET not in df.columns:
        raise ValueError(
            f"Target '{TARGET}' not found. "
            f"Available columns: {df.columns.tolist()}"
        )

    if df[TARGET].isna().any():
        raise ValueError("Target contains missing values.")

    if set(df[TARGET].unique()) != {0, 1}:
        raise ValueError(
            "Expected binary target values 0 and 1."
        )

    # Remove identifiers and failure-mode labels.
    leakage_columns = [
        "UDI",
        "Product_ID",
        "TWF",
        "HDF",
        "PWF",
        "OSF",
        "RNF",
    ]

    X = df.drop(
        columns=[TARGET] + leakage_columns,
        errors="ignore",
    )

    y = df[TARGET].astype(int)

    # Preserve the original class distribution in the test set.
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    print("\nOriginal target distribution:")
    print(y.value_counts().sort_index())

    print("\nTraining distribution before SMOTE:")
    print(y_train.value_counts().sort_index())

    print("\nTesting distribution (unchanged):")
    print(y_test.value_counts().sort_index())

    # SMOTE requires numeric features.
    # Encode the categorical Type feature before resampling.
    X_train_encoded = pd.get_dummies(
        X_train,
        columns=["Type"] if "Type" in X_train.columns else [],
        dtype=int,
    )

    X_test_encoded = pd.get_dummies(
        X_test,
        columns=["Type"] if "Type" in X_test.columns else [],
        dtype=int,
    )

    # Ensure the test columns match the training columns.
    X_test_encoded = X_test_encoded.reindex(
        columns=X_train_encoded.columns,
        fill_value=0,
    )

    # Fit SMOTE ONLY on the training set.
    smote = SMOTE(
        sampling_strategy="auto",
        random_state=RANDOM_STATE,
        k_neighbors=5,
    )

    X_train_balanced, y_train_balanced = (
        smote.fit_resample(X_train_encoded, y_train)
    )

    # Save training and test data separately.
    train_balanced = X_train_balanced.copy()
    train_balanced[TARGET] = y_train_balanced.to_numpy()

    test_original = X_test_encoded.copy()
    test_original[TARGET] = y_test.to_numpy()

    train_balanced.to_csv(
        OUTPUT_DIR / "train_balanced.csv",
        index=False,
    )

    test_original.to_csv(
        OUTPUT_DIR / "test_original.csv",
        index=False,
    )

    print("\nTraining distribution after SMOTE:")
    print(y_train_balanced.value_counts().sort_index())

    print("\nBalanced training shape:", train_balanced.shape)
    print("Original test shape:", test_original.shape)

    print("\nSaved:")
    print(OUTPUT_DIR / "train_balanced.csv")
    print(OUTPUT_DIR / "test_original.csv")


if __name__ == "__main__":
    main()
