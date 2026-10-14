
# src/tune_xgboost.py

from pathlib import Path
import json
import re

import joblib
import pandas as pd

from xgboost import XGBClassifier

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    average_precision_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)
from sklearn.model_selection import (
    train_test_split,
    RandomizedSearchCV,
    StratifiedKFold,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils.class_weight import compute_sample_weight


# --------------------------------------------------
# 1. Paths and configuration
# --------------------------------------------------

ROOT_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = ROOT_DIR / "data" / "raw" / "ai4i2020.csv"
MODEL_DIR = ROOT_DIR / "models"
REPORT_DIR = ROOT_DIR / "reports"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
TEST_SIZE = 0.20
N_ITER = 30


# --------------------------------------------------
# 2. Load dataset
# --------------------------------------------------

if not DATA_PATH.exists():
    raise FileNotFoundError(
        f"Dataset not found: {DATA_PATH}"
    )

df = pd.read_csv(DATA_PATH)

# Normalize column names.
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

TARGET = "machine_failure"

if TARGET not in df.columns:
    raise ValueError(
        f"Target '{TARGET}' not found. "
        f"Columns: {df.columns.tolist()}"
    )

df = df.dropna(subset=[TARGET]).copy()
y = df[TARGET].astype(int)

if not set(y.unique()).issubset({0, 1}):
    raise ValueError("Target must contain only 0 and 1.")

# Remove identifiers and failure-mode labels.
# These can leak information about the target.
columns_to_drop = [
    TARGET,
    "udi",
    "product_id",
    "twf",
    "hdf",
    "pwf",
    "osf",
    "rnf",
]

X = df.drop(
    columns=[
        col for col in columns_to_drop
        if col in df.columns
    ]
)

if X.empty:
    raise ValueError("No features remain after column removal.")

print("Dataset shape:", df.shape)
print("Features:", X.columns.tolist())
print("Target distribution:")
print(y.value_counts().sort_index())


# --------------------------------------------------
# 3. Split before fitting any preprocessing
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=TEST_SIZE,
    stratify=y,
    random_state=RANDOM_STATE,
)

print("\nTraining rows:", len(X_train))
print("Testing rows:", len(X_test))

print("\nTraining target distribution:")
print(y_train.value_counts().sort_index())

print("\nTesting target distribution:")
print(y_test.value_counts().sort_index())


# --------------------------------------------------
# 4. Build preprocessing pipeline
# --------------------------------------------------

numeric_features = X_train.select_dtypes(
    include="number"
).columns.tolist()

categorical_features = X_train.select_dtypes(
    exclude="number"
).columns.tolist()

numeric_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="median"),
        ),
    ]
)

categorical_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="most_frequent"),
        ),
        (
            "onehot",
            OneHotEncoder(handle_unknown="ignore"),
        ),
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        (
            "numeric",
            numeric_pipeline,
            numeric_features,
        ),
        (
            "categorical",
            categorical_pipeline,
            categorical_features,
        ),
    ],
    remainder="drop",
)


# --------------------------------------------------
# 5. Build complete ML pipeline
# --------------------------------------------------

xgb_model = XGBClassifier(
    objective="binary:logistic",
    eval_metric="logloss",
    tree_method="hist",
    random_state=RANDOM_STATE,
    n_jobs=1,
)

pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", xgb_model),
    ]
)


# --------------------------------------------------
# 6. Define hyperparameter search
# --------------------------------------------------

normal_count = int((y_train == 0).sum())
failure_count = int((y_train == 1).sum())

if failure_count == 0:
    raise ValueError("Training data contains no failures.")

imbalance_ratio = normal_count / failure_count

print("\nTraining imbalance ratio:", round(imbalance_ratio, 2))

param_distributions = {
    "model__n_estimators": [100, 200, 300, 400, 500],
    "model__max_depth": [2, 3, 4, 5, 6],
    "model__learning_rate": [
        0.01, 0.03, 0.05, 0.1, 0.15
    ],
    "model__min_child_weight": [1, 2, 3, 5, 7],
    "model__subsample": [0.7, 0.8, 0.9, 1.0],
    "model__colsample_bytree": [0.7, 0.8, 0.9, 1.0],
    "model__gamma": [0, 0.1, 0.2, 0.5, 1],
    "model__reg_alpha": [0, 0.01, 0.1, 1],
    "model__reg_lambda": [1, 2, 5, 10],
    "model__scale_pos_weight": [
        1, 5, 10, 20, round(imbalance_ratio, 2)
    ],
}

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=RANDOM_STATE,
)

search = RandomizedSearchCV(
    estimator=pipeline,
    param_distributions=param_distributions,
    n_iter=N_ITER,
    scoring="average_precision",
    cv=cv,
    refit=True,
    random_state=RANDOM_STATE,
    n_jobs=-1,
    verbose=2,
    return_train_score=True,
    error_score="raise",
)


# --------------------------------------------------
# 7. Fit with training data only
# --------------------------------------------------

# Compute weights separately inside each CV training fold.
# This avoids relying on pre-resampled data.
sample_weights = compute_sample_weight(
    class_weight="balanced",
    y=y_train,
)

print("\nStarting leakage-safe cross-validation...")
print("Folds:", cv.get_n_splits())
print("Configurations:", N_ITER)
print("Scoring: Average Precision (PR-AUC)")

search.fit(
    X_train,
    y_train,
    model__sample_weight=sample_weights,
)

best_pipeline = search.best_estimator_

print("\nTuning completed!")
print(
    "Best mean CV PR-AUC:",
    round(search.best_score_, 4),
)

print("\nBest parameters:")
print(search.best_params_)


# --------------------------------------------------
# 8. Save model and CV results
# --------------------------------------------------

model_path = MODEL_DIR / "xgboost_tuned.joblib"
joblib.dump(best_pipeline, model_path)

cv_results = pd.DataFrame(search.cv_results_)
cv_results = cv_results.sort_values("rank_test_score")

cv_results_path = REPORT_DIR / "xgboost_tuning_results.csv"
cv_results.to_csv(cv_results_path, index=False)

best_params_path = REPORT_DIR / "xgboost_best_params.json"

with open(best_params_path, "w", encoding="utf-8") as f:
    json.dump(
        {
            "best_cv_pr_auc": float(search.best_score_),
            "best_params": search.best_params_,
        },
        f,
        indent=4,
    )


# --------------------------------------------------
# 9. Evaluate on the held-out test set
# --------------------------------------------------

# The test set is not used for CV or hyperparameter selection.
y_pred = best_pipeline.predict(X_test)
y_prob = best_pipeline.predict_proba(X_test)[:, 1]

test_metrics = {
    "accuracy": float(accuracy_score(y_test, y_pred)),
    "precision": float(
        precision_score(y_test, y_pred, zero_division=0)
    ),
    "recall": float(
        recall_score(y_test, y_pred, zero_division=0)
    ),
    "f1_score": float(
        f1_score(y_test, y_pred, zero_division=0)
    ),
    "pr_auc_average_precision": float(
        average_precision_score(y_test, y_prob)
    ),
    "roc_auc": float(
        roc_auc_score(y_test, y_prob)
    ),
}

print("\nFinal test metrics (threshold = 0.5):")

for name, value in test_metrics.items():
    print(f"{name}: {value:.4f}")

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred,
        labels=[0, 1],
        target_names=["Normal", "Failure"],
        zero_division=0,
    )
)


# --------------------------------------------------
# 10. Save test metrics and summary
# --------------------------------------------------

metrics_path = (
    REPORT_DIR / "xgboost_tuned_test_metrics.json"
)

with open(metrics_path, "w", encoding="utf-8") as f:
    json.dump(test_metrics, f, indent=4)

summary_path = REPORT_DIR / "xgboost_training_summary.json"

with open(summary_path, "w", encoding="utf-8") as f:
    json.dump(
        {
            "dataset": DATA_PATH.name,
            "training_rows": len(X_train),
            "test_rows": len(X_test),
            "input_features": X.columns.tolist(),
            "cv_folds": cv.get_n_splits(),
            "search_iterations": N_ITER,
            "best_cv_pr_auc": float(search.best_score_),
            "best_params": search.best_params_,
            "test_metrics": test_metrics,
        },
        f,
        indent=4,
    )

print("\nSaved files:")
print("Model:", model_path)
print("CV results:", cv_results_path)
print("Best parameters:", best_params_path)
print("Test metrics:", metrics_path)
print("Training summary:", summary_path)

print("\nDone!")


