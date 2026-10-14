
import json
import re
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import shap

from sklearn.model_selection import train_test_split


ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = ROOT / "data" / "raw" / "ai4i2020.csv"
MODEL_PATH = ROOT / "models" / "final_model.joblib"
THRESHOLD_PATH = ROOT / "reports" / "final_decision_threshold.json"
METRICS_PATH = ROOT / "reports" / "final_test_metrics.json"

REPORTS = ROOT / "reports"
REPORTS.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
MAX_ROWS = 500


def normalize_column_name(name):
    name = name.strip().lower()
    name = re.sub(r"[^a-z0-9]+", "_", name)
    return name.strip("_")


print("Loading dataset and final model...")

df = pd.read_csv(DATA_PATH)
df.columns = [normalize_column_name(c) for c in df.columns]

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

X = df.drop(columns=excluded)
y = df["machine_failure"].astype(int)

# Reproduce the exact split used in final_train.py.
X_development, X_test, y_development, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    stratify=y,
    random_state=RANDOM_STATE,
)

X_train, X_validation, y_train, y_validation = train_test_split(
    X_development,
    y_development,
    test_size=0.25,
    stratify=y_development,
    random_state=RANDOM_STATE,
)

pipeline = joblib.load(MODEL_PATH)

with open(THRESHOLD_PATH, encoding="utf-8") as f:
    threshold_config = json.load(f)

threshold = float(threshold_config["threshold"])

with open(METRICS_PATH, encoding="utf-8") as f:
    metrics = json.load(f)

model_name = metrics["selected_model"]

print(f"Selected model: {model_name}")
print(f"Decision threshold: {threshold}")
print(f"Test rows: {len(X_test)}")

# Use the fitted preprocessing pipeline.
preprocessor = pipeline.named_steps["preprocessor"]
estimator = pipeline.named_steps["model"]

X_transformed = preprocessor.transform(X_test)

feature_names = preprocessor.get_feature_names_out()

X_transformed = pd.DataFrame(
    X_transformed,
    columns=feature_names,
    index=X_test.index,
)

sample_size = min(MAX_ROWS, len(X_transformed))
X_explain = X_transformed.sample(
    n=sample_size,
    random_state=RANDOM_STATE,
)

print(f"Calculating SHAP values for {sample_size} test rows...")

# TreeExplainer for tree-based models; generic model-agnostic
# explainer for Logistic Regression.
if model_name in ["XGBoost", "Random Forest"]:
    explainer = shap.TreeExplainer(estimator)
    shap_values = explainer(X_explain)
else:
    # Use a small background sample for efficiency.
    background = X_transformed.sample(
        n=min(50, len(X_transformed)),
        random_state=RANDOM_STATE,
    )
    explainer = shap.Explainer(
        estimator.predict_proba,
        background,
        algorithm="permutation",
    )
    shap_values = explainer(
        X_explain,
        max_evals=2 * X_explain.shape[1] + 1,
    )

# Select the positive class for binary classification
# when SHAP returns class-specific values.
if len(shap_values.shape) == 3:
    shap_values = shap_values[:, :, 1]

print("Saving SHAP summary plot...")

shap.plots.beeswarm(
    shap_values,
    max_display=15,
    show=False,
)

plt.title(f"{model_name} - SHAP Summary")
plt.tight_layout()
plt.savefig(
    REPORTS / "final_shap_summary.png",
    dpi=300,
    bbox_inches="tight",
)
plt.close()

print("Saving SHAP importance plot...")

shap.plots.bar(
    shap_values,
    max_display=15,
    show=False,
)

plt.title(f"{model_name} - Mean Absolute SHAP Importance")
plt.tight_layout()
plt.savefig(
    REPORTS / "final_shap_feature_importance.png",
    dpi=300,
    bbox_inches="tight",
)
plt.close()

# Explain a correctly detected failure if one exists.
probabilities = pipeline.predict_proba(X_test)[:, 1]
predictions = (probabilities >= threshold).astype(int)

detected = [
    i for i, (actual, predicted) in enumerate(
        zip(y_test.to_numpy(), predictions)
    )
    if actual == 1 and predicted == 1
]

chosen = detected[0] if detected else int(probabilities.argmax())
test_index = X_test.index[chosen]

# Recalculate the explanation for this exact test row.
one_row = X_transformed.loc[[test_index]]

if model_name in ["XGBoost", "Random Forest"]:
    one_explanation = explainer(one_row)
else:
    one_explanation = explainer(
        one_row,
        max_evals=2 * one_row.shape[1] + 1,
    )

if len(one_explanation.shape) == 3:
    one_explanation = one_explanation[:, :, 1]

shap.plots.waterfall(
    one_explanation[0],
    max_display=15,
    show=False,
)

plt.title("Final Model - Individual Prediction Explanation")
plt.tight_layout()
plt.savefig(
    REPORTS / "final_shap_waterfall.png",
    dpi=300,
    bbox_inches="tight",
)
plt.close()

importance = pd.DataFrame({
    "feature": list(shap_values.feature_names),
    "mean_absolute_shap": abs(shap_values.values).mean(axis=0),
}).sort_values("mean_absolute_shap", ascending=False)

importance.to_csv(
    REPORTS / "final_shap_feature_importance.csv",
    index=False,
)

example = {
    "model": model_name,
    "test_row_index": int(test_index),
    "actual_failure": int(y_test.loc[test_index]),
    "predicted_failure": int(predictions[chosen]),
    "failure_probability": float(probabilities[chosen]),
    "decision_threshold": threshold,
}

with open(
    REPORTS / "final_shap_example.json",
    "w",
    encoding="utf-8",
) as f:
    json.dump(example, f, indent=2)

print("\nTop 10 SHAP features:")
print(importance.head(10).to_string(index=False))

print("\nExample prediction:")
print(json.dumps(example, indent=2))

print("\nStep 1 completed.")
print("Generated files:")
print(REPORTS / "final_shap_summary.png")
print(REPORTS / "final_shap_feature_importance.png")
print(REPORTS / "final_shap_waterfall.png")
print(REPORTS / "final_shap_feature_importance.csv")
print(REPORTS / "final_shap_example.json")
