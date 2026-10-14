
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


# --------------------------------------------------
# 1. Project paths and configuration
# --------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = ROOT_DIR / "data" / "raw" / "ai4i2020.csv"
MODEL_PATH = ROOT_DIR / "models" / "xgboost_tuned.joblib"
THRESHOLD_PATH = ROOT_DIR / "reports" / "decision_threshold.json"

REPORTS_DIR = ROOT_DIR / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
MAX_EXPLANATION_ROWS = 500


# --------------------------------------------------
# 2. Load and prepare dataset
# --------------------------------------------------
def normalize_column_name(name):
    name = name.strip().lower()
    name = re.sub(r"[^a-z0-9]+", "_", name)
    return name.strip("_")


print("Loading dataset...")

df = pd.read_csv(DATA_PATH)
df.columns = [normalize_column_name(c) for c in df.columns]

target = "machine_failure"

excluded_columns = [
    "udi",
    "product_id",
    "machine_failure",
    "twf",
    "hdf",
    "pwf",
    "osf",
    "rnf",
]

X = df.drop(columns=excluded_columns)
y = df[target]

# Reproduce the original train/test split.
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    stratify=y,
    random_state=RANDOM_STATE,
)

print(f"Dataset rows: {len(df)}")
print(f"Test rows: {len(X_test)}")
print(f"Predictor columns: {list(X.columns)}")


# --------------------------------------------------
# 3. Load trained pipeline and threshold
# --------------------------------------------------
print("\nLoading trained model...")

pipeline = joblib.load(MODEL_PATH)

with open(THRESHOLD_PATH, "r", encoding="utf-8") as file:
    threshold_config = json.load(file)

threshold = float(threshold_config["threshold"])

print(f"Decision threshold: {threshold}")


# --------------------------------------------------
# 4. Apply saved preprocessing
# --------------------------------------------------
preprocessor = pipeline.named_steps["preprocessor"]
xgb_model = pipeline.named_steps["model"]

# Use the fitted preprocessing pipeline.
X_transformed = preprocessor.transform(X_test)

try:
    feature_names = preprocessor.get_feature_names_out()
except (AttributeError, ValueError):
    feature_names = [
        f"feature_{i}" for i in range(X_transformed.shape[1])
    ]

# Convert to a DataFrame for readable SHAP plots.
X_transformed_df = pd.DataFrame(
    X_transformed,
    columns=feature_names,
    index=X_test.index,
)

# Explain a representative sample of the test data.
sample_size = min(MAX_EXPLANATION_ROWS, len(X_transformed_df))

X_explain = X_transformed_df.sample(
    n=sample_size,
    random_state=RANDOM_STATE,
)

print(f"Rows used for SHAP summary: {len(X_explain)}")


# --------------------------------------------------
# 5. Generate SHAP explanations
# --------------------------------------------------
print("\nCalculating SHAP values...")

explainer = shap.TreeExplainer(xgb_model)
shap_values = explainer(X_explain)

print("SHAP values calculated successfully.")


# --------------------------------------------------
# 6. Global SHAP summary plot
# --------------------------------------------------
print("\nCreating SHAP summary plot...")

plt.figure(figsize=(11, 7))

shap.plots.beeswarm(
    shap_values,
    max_display=15,
    show=False,
)

plt.title("SHAP Summary - Predictive Maintenance")
plt.tight_layout()

summary_path = REPORTS_DIR / "shap_summary.png"
plt.savefig(summary_path, dpi=300, bbox_inches="tight")
plt.close()


# --------------------------------------------------
# 7. Global feature importance bar plot
# --------------------------------------------------
print("Creating feature importance plot...")

plt.figure(figsize=(10, 7))

shap.plots.bar(
    shap_values,
    max_display=15,
    show=False,
)

plt.title("Mean Absolute SHAP Feature Importance")
plt.tight_layout()

importance_path = REPORTS_DIR / "shap_feature_importance.png"
plt.savefig(importance_path, dpi=300, bbox_inches="tight")
plt.close()


# --------------------------------------------------
# 8. Explain one individual prediction
# --------------------------------------------------
print("Creating individual prediction explanation...")

test_probabilities = pipeline.predict_proba(X_test)[:, 1]
test_predictions = (test_probabilities >= threshold).astype(int)

# Prefer an actual failure correctly detected by the model.
positive_indices = [
    i for i, (actual, predicted) in enumerate(
        zip(y_test.to_numpy(), test_predictions)
    )
    if actual == 1 and predicted == 1
]

# Otherwise, explain the first predicted failure.
if positive_indices:
    chosen_index = positive_indices[0]
else:
    predicted_indices = [
        i for i, predicted in enumerate(test_predictions)
        if predicted == 1
    ]

    if predicted_indices:
        chosen_index = predicted_indices[0]
    else:
        chosen_index = int(test_probabilities.argmax())

chosen_row_index = X_test.index[chosen_index]
row_position = X_explain.index.get_indexer([chosen_row_index])[0]

if row_position >= 0:
    explanation = shap_values[row_position]

    plt.figure(figsize=(10, 7))

    shap.plots.waterfall(
        explanation,
        max_display=15,
        show=False,
    )

    plt.title("SHAP Explanation for One Machine")
    plt.tight_layout()

    waterfall_path = REPORTS_DIR / "shap_waterfall_example.png"
    plt.savefig(waterfall_path, dpi=300, bbox_inches="tight")
    plt.close()
else:
    # Ensure the selected individual row is included in the explanation.
    single_row = X_transformed_df.loc[[chosen_row_index]]
    single_explanation = explainer(single_row)

    plt.figure(figsize=(10, 7))

    shap.plots.waterfall(
        single_explanation[0],
        max_display=15,
        show=False,
    )

    plt.title("SHAP Explanation for One Machine")
    plt.tight_layout()

    waterfall_path = REPORTS_DIR / "shap_waterfall_example.png"
    plt.savefig(waterfall_path, dpi=300, bbox_inches="tight")
    plt.close()


# --------------------------------------------------
# 9. Export mean absolute SHAP importance
# --------------------------------------------------
importance = pd.DataFrame({
    "feature": list(feature_names),
    "mean_absolute_shap": abs(shap_values.values).mean(axis=0),
})

importance = importance.sort_values(
    "mean_absolute_shap",
    ascending=False,
)

importance.to_csv(
    REPORTS_DIR / "shap_feature_importance.csv",
    index=False,
)

# Save the example prediction details.
example_result = {
    "test_row_index": int(chosen_row_index),
    "actual_machine_failure": int(y_test.loc[chosen_row_index]),
    "predicted_machine_failure": int(test_predictions[chosen_index]),
    "predicted_failure_probability": float(
        test_probabilities[chosen_index]
    ),
    "decision_threshold": threshold,
}

with open(
    REPORTS_DIR / "shap_example_prediction.json",
    "w",
    encoding="utf-8",
) as file:
    json.dump(example_result, file, indent=2)


# --------------------------------------------------
# 10. Print results
# --------------------------------------------------
print("\nTop 10 SHAP features:")
print(importance.head(10).to_string(index=False))

print("\nExample prediction:")
print(json.dumps(example_result, indent=2))

print("\nSHAP explainability completed successfully.")
print(f"Summary plot: {summary_path}")
print(f"Importance plot: {importance_path}")
print(f"Waterfall plot: {waterfall_path}")
print(
    "Importance CSV: "
    f"{REPORTS_DIR / 'shap_feature_importance.csv'}"
)
print(
    "Example JSON: "
    f"{REPORTS_DIR / 'shap_example_prediction.json'}"
)
