
# src/select_threshold.py

from pathlib import Path
import json
import re

import joblib
import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    average_precision_score,
    roc_auc_score,
    confusion_matrix,
)

ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = ROOT / "data" / "raw" / "ai4i2020.csv"
MODEL_PATH = ROOT / "models" / "xgboost_tuned.joblib"
REPORT_DIR = ROOT / "reports"

REPORT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
TEST_SIZE = 0.20
VALIDATION_SIZE = 0.20


# 1. Load and prepare the original dataset
df = pd.read_csv(DATA_PATH)

df.columns = [
    re.sub(
        r"_+",
        "_",
        re.sub(
            r"[^a-zA-Z0-9]+",
            "_",
            col.strip(),
        ).strip("_"),
    ).lower()
    for col in df.columns
]

df = df.dropna(subset=["machine_failure"]).copy()

y = df["machine_failure"].astype(int)

drop_columns = [
    "machine_failure",
    "udi",
    "product_id",
    "twf",
    "hdf",
    "pwf",
    "osf",
    "rnf",
]

X = df.drop(
    columns=[c for c in drop_columns if c in df.columns]
)


# 2. Recreate the same held-out test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=TEST_SIZE,
    stratify=y,
    random_state=RANDOM_STATE,
)

# Split training data into model-fitting and validation sets.
X_fit, X_val, y_fit, y_val = train_test_split(
    X_train,
    y_train,
    test_size=VALIDATION_SIZE,
    stratify=y_train,
    random_state=RANDOM_STATE,
)

print("Model-fitting rows:", len(X_fit))
print("Validation rows:", len(X_val))
print("Test rows:", len(X_test))


# 3. Load the selected pipeline and refit on model-fitting data
if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Model not found: {MODEL_PATH}"
    )

saved_pipeline = joblib.load(MODEL_PATH)
validation_model = clone(saved_pipeline)

sample_weights = compute_sample_weight(
    class_weight="balanced",
    y=y_fit,
)

validation_model.fit(
    X_fit,
    y_fit,
    model__sample_weight=sample_weights,
)


# 4. Obtain validation probabilities
val_prob = validation_model.predict_proba(X_val)[:, 1]

# Evaluate several candidate thresholds.
thresholds = np.arange(0.10, 0.81, 0.05)

rows = []

for threshold in thresholds:
    val_pred = (val_prob >= threshold).astype(int)

    rows.append(
        {
            "threshold": round(float(threshold), 2),
            "precision": precision_score(
                y_val, val_pred, zero_division=0
            ),
            "recall": recall_score(
                y_val, val_pred, zero_division=0
            ),
            "f1_score": f1_score(
                y_val, val_pred, zero_division=0
            ),
            "false_alarms": int(
                ((val_pred == 1) & (y_val.to_numpy() == 0)).sum()
            ),
            "missed_failures": int(
                ((val_pred == 0) & (y_val.to_numpy() == 1)).sum()
            ),
        }
    )

results = pd.DataFrame(rows)

results_path = REPORT_DIR / "threshold_validation_results.csv"
results.to_csv(results_path, index=False)

print("\nValidation results:")
print(results.round(4).to_string(index=False))


# 5. Select the threshold with the highest validation F1-score
# This balances precision and recall.
best_index = results["f1_score"].idxmax()
best_row = results.loc[best_index]

best_threshold = float(best_row["threshold"])

print("\nSelected threshold:", best_threshold)
print("Validation precision:", round(best_row["precision"], 4))
print("Validation recall:", round(best_row["recall"], 4))
print("Validation F1-score:", round(best_row["f1_score"], 4))


# 6. Save the selected threshold
threshold_path = REPORT_DIR / "decision_threshold.json"

with open(threshold_path, "w", encoding="utf-8") as f:
    json.dump(
        {
            "threshold": best_threshold,
            "selection_method": "Highest validation F1-score",
            "validation_precision": float(best_row["precision"]),
            "validation_recall": float(best_row["recall"]),
            "validation_f1_score": float(best_row["f1_score"]),
        },
        f,
        indent=4,
    )


# 7. Final evaluation on the test set
# The threshold was chosen without using test labels.
test_prob = validation_model.predict_proba(X_test)[:, 1]
test_pred = (test_prob >= best_threshold).astype(int)

print("\nTest metrics at selected threshold:")
print("Precision:", round(
    precision_score(y_test, test_pred, zero_division=0), 4
))
print("Recall:", round(
    recall_score(y_test, test_pred, zero_division=0), 4
))
print("F1-score:", round(
    f1_score(y_test, test_pred, zero_division=0), 4
))
print("PR-AUC:", round(
    average_precision_score(y_test, test_prob), 4
))
print("ROC-AUC:", round(
    roc_auc_score(y_test, test_prob), 4
))

print("\nConfusion matrix:")
print(confusion_matrix(y_test, test_pred))

print("\nSaved:")
print(results_path)
print(threshold_path)


