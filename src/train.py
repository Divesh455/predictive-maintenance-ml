
from pathlib import Path

import joblib
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    average_precision_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay,
)

DATA_DIR = Path("data/processed")
MODEL_DIR = Path("models")
REPORT_DIR = Path("reports")
FIGURE_DIR = REPORT_DIR / "figures"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

TARGET = "Machine_failure"
RANDOM_STATE = 42


def main():
    train_path = DATA_DIR / "train_balanced.csv"
    test_path = DATA_DIR / "test_original.csv"

    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError(
            "Training or test CSV not found. "
            "Run python -m src.balance_data first."
        )

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    X_train = train_df.drop(columns=[TARGET])
    y_train = train_df[TARGET].astype(int)

    X_test = test_df.drop(columns=[TARGET])
    y_test = test_df[TARGET].astype(int)

    # Keep feature order consistent.
    X_test = X_test.reindex(
        columns=X_train.columns,
        fill_value=0,
    )

    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=2000,
            random_state=RANDOM_STATE,
        ),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=10,
            min_samples_leaf=5,
            random_state=RANDOM_STATE,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "XGBoost": XGBClassifier(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            eval_metric="logloss",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
    }

    results = []
    trained_models = {}

    for name, model in models.items():
        print(f"\nTraining: {name}")
        model.fit(X_train, y_train)

        # Default threshold of 0.5 for this initial comparison.
        predictions = model.predict(X_test)

        probabilities = model.predict_proba(X_test)[:, 1]

        metrics = {
            "Model": name,
            "Accuracy": accuracy_score(
                y_test, predictions
            ),
            "Precision": precision_score(
                y_test, predictions, zero_division=0
            ),
            "Recall": recall_score(
                y_test, predictions, zero_division=0
            ),
            "F1_Score": f1_score(
                y_test, predictions, zero_division=0
            ),
            "PR_AUC": average_precision_score(
                y_test, probabilities
            ),
            "ROC_AUC": roc_auc_score(
                y_test, probabilities
            ),
        }

        results.append(metrics)
        trained_models[name] = model

        print(pd.Series(metrics).to_string())
        print("\nClassification report:")
        print(classification_report(
            y_test,
            predictions,
            zero_division=0,
        ))

        # Save a confusion matrix for each model.
        display = ConfusionMatrixDisplay(
            confusion_matrix=confusion_matrix(
                y_test, predictions
            ),
            display_labels=["Normal", "Failure"],
        )

        display.plot(values_format="d")
        plt.title(f"{name} - Confusion Matrix")
        plt.tight_layout()

        filename = (
            name.lower().replace(" ", "_")
            + "_confusion_matrix.png"
        )

        plt.savefig(
            FIGURE_DIR / filename,
            dpi=150,
        )
        plt.close()

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values(
        by="PR_AUC",
        ascending=False,
    )

    results_df.to_csv(
        REPORT_DIR / "model_comparison.csv",
        index=False,
    )

    print("\n=== MODEL COMPARISON ===")
    print(results_df.to_string(index=False))

    print(
        "\nResults saved to:",
        REPORT_DIR / "model_comparison.csv",
    )


if __name__ == "__main__":
    main()
