
import json
from pathlib import Path
import re

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    StratifiedKFold,
    cross_validate,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier


# ==================================================
# 1. Configuration
# ==================================================
ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = ROOT / "data" / "raw" / "ai4i2020.csv"
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
TEST_SIZE = 0.20
VALIDATION_SIZE_OF_DEVELOPMENT = 0.25
CV_FOLDS = 5

# Threshold candidates for validation-based F1 selection.
THRESHOLDS = np.round(np.arange(0.10, 0.901, 0.01), 2)


# ==================================================
# 2. Load and validate data
# ==================================================
def normalize_column_name(name):
    name = name.strip().lower()
    name = re.sub(r"[^a-z0-9]+", "_", name)
    return name.strip("_")


print("=" * 65)
print("FINAL PREDICTIVE MAINTENANCE MODEL TRAINING")
print("=" * 65)

df = pd.read_csv(DATA_PATH)
df.columns = [normalize_column_name(c) for c in df.columns]

target = "machine_failure"

excluded = [
    "udi",
    "product_id",
    "machine_failure",
    "twf",
    "hdf",
    "pwf",
    "osf",
    "rnf",
]

required = {
    target,
    "type",
    "air_temperature_k",
    "process_temperature_k",
    "rotational_speed_rpm",
    "torque_nm",
    "tool_wear_min",
}

missing = required - set(df.columns)
if missing:
    raise ValueError(f"Required columns are missing: {sorted(missing)}")

X = df.drop(columns=excluded, errors="ignore")
y = df[target].astype(int)

if set(y.unique()) != {0, 1}:
    raise ValueError("machine_failure must contain both 0 and 1.")

categorical_features = ["type"]
numeric_features = [c for c in X.columns if c not in categorical_features]

print(f"\nDataset shape: {df.shape}")
print(f"Predictors: {list(X.columns)}")
print(f"Failure rate: {y.mean():.2%}")


# ==================================================
# 3. Create untouched test set
# ==================================================
# Test labels are not used for CV, model selection,
# threshold selection, or fitting.
X_development, X_test, y_development, y_test = train_test_split(
    X,
    y,
    test_size=TEST_SIZE,
    stratify=y,
    random_state=RANDOM_STATE,
)

# Development set: 80% of data.
# Training: 60% of total data.
# Validation: 20% of total data.
X_train, X_validation, y_train, y_validation = train_test_split(
    X_development,
    y_development,
    test_size=VALIDATION_SIZE_OF_DEVELOPMENT,
    stratify=y_development,
    random_state=RANDOM_STATE,
)

print("\nData split:")
print(f"Training rows:   {len(X_train)}")
print(f"Validation rows: {len(X_validation)}")
print(f"Test rows:       {len(X_test)}")
print("\nThe test set will remain untouched until final evaluation.")


# ==================================================
# 4. Build preprocessing pipeline
# ==================================================
# All preprocessing is fitted separately inside each
# cross-validation fold by the sklearn Pipeline.
numeric_preprocessor = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])

categorical_preprocessor = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore")),
])

preprocessor = ColumnTransformer([
    ("numeric", numeric_preprocessor, numeric_features),
    ("categorical", categorical_preprocessor, categorical_features),
])


# ==================================================
# 5. Define candidate models
# ==================================================
negative_count = int((y_train == 0).sum())
positive_count = int((y_train == 1).sum())
scale_pos_weight = negative_count / positive_count

models = {
    "Logistic Regression": LogisticRegression(
        class_weight="balanced",
        max_iter=3000,
        random_state=RANDOM_STATE,
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=400,
        max_depth=None,
        min_samples_leaf=2,
        class_weight="balanced_subsample",
        random_state=RANDOM_STATE,
        n_jobs=1,
    ),

    "XGBoost": XGBClassifier(
        n_estimators=400,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=1.0,
        min_child_weight=1,
        gamma=0,
        reg_alpha=0.01,
        reg_lambda=1,
        scale_pos_weight=scale_pos_weight,
        eval_metric="logloss",
        random_state=RANDOM_STATE,
        n_jobs=1,
        verbosity=0,
    ),
}


# ==================================================
# 6. Compare models using training-only CV
# ==================================================
cv = StratifiedKFold(
    n_splits=CV_FOLDS,
    shuffle=True,
    random_state=RANDOM_STATE,
)

cv_results = []
fitted_training_models = {}

print("\n" + "=" * 65)
print("CROSS-VALIDATION MODEL COMPARISON")
print("=" * 65)

for model_name, estimator in models.items():
    print(f"\nEvaluating {model_name}...")

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", estimator),
    ])

    scores = cross_validate(
        pipeline,
        X_train,
        y_train,
        cv=cv,
        scoring={
            "f1": "f1",
            "precision": "precision",
            "recall": "recall",
            "average_precision": "average_precision",
            "roc_auc": "roc_auc",
        },
        return_train_score=False,
        n_jobs=-1,
        error_score="raise",
    )

    result = {
        "model": model_name,
        "cv_f1_mean": float(scores["test_f1"].mean()),
        "cv_f1_std": float(scores["test_f1"].std()),
        "cv_precision_mean": float(scores["test_precision"].mean()),
        "cv_recall_mean": float(scores["test_recall"].mean()),
        "cv_pr_auc_mean": float(
            scores["test_average_precision"].mean()
        ),
        "cv_roc_auc_mean": float(scores["test_roc_auc"].mean()),
    }

    cv_results.append(result)

    print(f"Mean CV F1:        {result['cv_f1_mean']:.4f}")
    print(f"Mean CV precision: {result['cv_precision_mean']:.4f}")
    print(f"Mean CV recall:    {result['cv_recall_mean']:.4f}")
    print(f"Mean CV PR-AUC:    {result['cv_pr_auc_mean']:.4f}")

comparison = pd.DataFrame(cv_results).sort_values(
    ["cv_f1_mean", "cv_pr_auc_mean"],
    ascending=False,
)

comparison.to_csv(
    REPORT_DIR / "final_model_comparison_cv.csv",
    index=False,
)

print("\nCV comparison:")
print(comparison.to_string(index=False))

# Model selection is based only on training CV F1.
best_model_name = comparison.iloc[0]["model"]
print(f"\nSelected model: {best_model_name}")


# ==================================================
# 7. Fit selected model on training set only
# ==================================================
selected_estimator = models[best_model_name]

final_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", selected_estimator),
])

print("\nFitting selected model on training rows...")
final_pipeline.fit(X_train, y_train)


# ==================================================
# 8. Select threshold using validation set only
# ==================================================
print("\nSelecting decision threshold using validation data...")

validation_probabilities = final_pipeline.predict_proba(
    X_validation
)[:, 1]

threshold_results = []

for threshold in THRESHOLDS:
    validation_predictions = (
        validation_probabilities >= threshold
    ).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_validation,
        validation_predictions,
        labels=[0, 1],
    ).ravel()

    threshold_results.append({
        "threshold": float(threshold),
        "precision": float(
            precision_score(
                y_validation,
                validation_predictions,
                zero_division=0,
            )
        ),
        "recall": float(
            recall_score(
                y_validation,
                validation_predictions,
                zero_division=0,
            )
        ),
        "f1_score": float(
            f1_score(
                y_validation,
                validation_predictions,
                zero_division=0,
            )
        ),
        "false_alarms": int(fp),
        "missed_failures": int(fn),
        "true_negatives": int(tn),
        "true_positives": int(tp),
    })

threshold_df = pd.DataFrame(threshold_results)

threshold_df.to_csv(
    REPORT_DIR / "final_threshold_validation_results.csv",
    index=False,
)

# If F1 ties, prefer the threshold closer to 0.50.
threshold_df["distance_from_050"] = (
    threshold_df["threshold"] - 0.50
).abs()

threshold_df = threshold_df.sort_values(
    ["f1_score", "distance_from_050"],
    ascending=[False, True],
)

best_threshold_row = threshold_df.iloc[0]
best_threshold = float(best_threshold_row["threshold"])

threshold_config = {
    "threshold": best_threshold,
    "selection_method": (
        "Highest validation F1-score; ties prefer threshold closest to 0.50"
    ),
    "validation_precision": float(best_threshold_row["precision"]),
    "validation_recall": float(best_threshold_row["recall"]),
    "validation_f1_score": float(best_threshold_row["f1_score"]),
    "selected_model": best_model_name,
}

with open(
    REPORT_DIR / "final_decision_threshold.json",
    "w",
    encoding="utf-8",
) as file:
    json.dump(threshold_config, file, indent=2)

print(f"\nSelected threshold: {best_threshold:.2f}")
print(f"Validation precision: {best_threshold_row['precision']:.4f}")
print(f"Validation recall:    {best_threshold_row['recall']:.4f}")
print(f"Validation F1-score:  {best_threshold_row['f1_score']:.4f}")


# ==================================================
# 9. Evaluate once on untouched test set
# ==================================================
print("\n" + "=" * 65)
print("FINAL UNTOUCHED TEST EVALUATION")
print("=" * 65)

test_probabilities = final_pipeline.predict_proba(X_test)[:, 1]
test_predictions = (test_probabilities >= best_threshold).astype(int)

tn, fp, fn, tp = confusion_matrix(
    y_test,
    test_predictions,
    labels=[0, 1],
).ravel()

test_metrics = {
    "selected_model": best_model_name,
    "decision_threshold": best_threshold,
    "test_rows": int(len(X_test)),
    "accuracy": float(accuracy_score(y_test, test_predictions)),
    "precision": float(
        precision_score(y_test, test_predictions, zero_division=0)
    ),
    "recall": float(
        recall_score(y_test, test_predictions, zero_division=0)
    ),
    "f1_score": float(
        f1_score(y_test, test_predictions, zero_division=0)
    ),
    "pr_auc": float(
        average_precision_score(y_test, test_probabilities)
    ),
    "roc_auc": float(
        roc_auc_score(y_test, test_probabilities)
    ),
    "confusion_matrix": {
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
    },
}

print("\nFinal test metrics:")
for key, value in test_metrics.items():
    if key != "confusion_matrix":
        print(f"{key}: {value}")

print("\nConfusion matrix [[TN, FP], [FN, TP]]:")
print(confusion_matrix(y_test, test_predictions, labels=[0, 1]))

print("\nClassification report:")
print(
    classification_report(
        y_test,
        test_predictions,
        target_names=["Normal", "Failure"],
        zero_division=0,
    )
)

with open(
    REPORT_DIR / "final_test_metrics.json",
    "w",
    encoding="utf-8",
) as file:
    json.dump(test_metrics, file, indent=2)

# Save the confusion matrix plot.
fig, ax = plt.subplots(figsize=(6, 5))
matrix = confusion_matrix(y_test, test_predictions, labels=[0, 1])

image = ax.imshow(matrix, cmap="Blues")
ax.set_title(f"{best_model_name} - Final Test Confusion Matrix")
ax.set_xlabel("Predicted label")
ax.set_ylabel("Actual label")
ax.set_xticks([0, 1], labels=["Normal", "Failure"])
ax.set_yticks([0, 1], labels=["Normal", "Failure"])

for row in range(2):
    for col in range(2):
        ax.text(
            col,
            row,
            str(matrix[row, col]),
            ha="center",
            va="center",
            color="black",
        )

fig.colorbar(image, ax=ax)
fig.tight_layout()
fig.savefig(
    REPORT_DIR / "final_test_confusion_matrix.png",
    dpi=300,
    bbox_inches="tight",
)
plt.close(fig)


# ==================================================
# 10. Save final fitted model and metadata
# ==================================================
model_path = MODEL_DIR / "final_model.joblib"
joblib.dump(final_pipeline, model_path)

metadata = {
    "model_name": best_model_name,
    "target": target,
    "predictor_columns": list(X.columns),
    "categorical_features": categorical_features,
    "numeric_features": numeric_features,
    "decision_threshold": best_threshold,
    "random_state": RANDOM_STATE,
    "training_rows": int(len(X_train)),
    "validation_rows": int(len(X_validation)),
    "test_rows": int(len(X_test)),
    "cv_folds": CV_FOLDS,
    "selection_metric": "Mean training cross-validation F1-score",
    "threshold_selection_metric": "Validation F1-score",
    "test_metrics_file": "reports/final_test_metrics.json",
}

with open(
    REPORT_DIR / "final_model_metadata.json",
    "w",
    encoding="utf-8",
) as file:
    json.dump(metadata, file, indent=2)

print("\n" + "=" * 65)
print("FINAL MODEL ARTIFACTS SAVED")
print("=" * 65)
print(f"Model: {model_path}")
print(f"CV comparison: {REPORT_DIR / 'final_model_comparison_cv.csv'}")
print(f"Threshold config: {REPORT_DIR / 'final_decision_threshold.json'}")
print(f"Test metrics: {REPORT_DIR / 'final_test_metrics.json'}")
print(
    "Confusion matrix: "
    f"{REPORT_DIR / 'final_test_confusion_matrix.png'}"
)
print(f"Metadata: {REPORT_DIR / 'final_model_metadata.json'}")
print("\nTraining workflow completed.")
