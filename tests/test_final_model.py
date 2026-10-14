
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = ROOT / "models" / "final_model.joblib"
THRESHOLD_PATH = ROOT / "reports" / "final_decision_threshold.json"
METADATA_PATH = ROOT / "reports" / "final_model_metadata.json"


@pytest.fixture(scope="module")
def model():
    assert MODEL_PATH.exists(), f"Model not found: {MODEL_PATH}"
    return joblib.load(MODEL_PATH)


@pytest.fixture(scope="module")
def threshold():
    assert THRESHOLD_PATH.exists(), "Threshold configuration not found"

    with open(THRESHOLD_PATH, encoding="utf-8") as file:
        config = json.load(file)

    return float(config["threshold"])


@pytest.fixture(scope="module")
def metadata():
    assert METADATA_PATH.exists(), "Model metadata not found"

    with open(METADATA_PATH, encoding="utf-8") as file:
        return json.load(file)


@pytest.fixture
def valid_input():
    return pd.DataFrame([{
        "type": "M",
        "air_temperature_k": 300.0,
        "process_temperature_k": 310.0,
        "rotational_speed_rpm": 1500,
        "torque_nm": 40.0,
        "tool_wear_min": 100,
    }])


def test_model_loads(model):
    assert model is not None
    assert hasattr(model, "predict_proba")


def test_expected_features(metadata, valid_input):
    expected = metadata["predictor_columns"]

    assert list(valid_input.columns) == expected


def test_probability_is_valid(model, valid_input):
    probabilities = model.predict_proba(valid_input)

    assert probabilities.shape == (1, 2)
    assert np.isfinite(probabilities).all()
    assert np.all(probabilities >= 0)
    assert np.all(probabilities <= 1)
    assert probabilities.sum(axis=1)[0] == pytest.approx(1.0)


def test_prediction_is_binary(model, valid_input, threshold):
    probability = model.predict_proba(valid_input)[0, 1]
    prediction = int(probability >= threshold)

    assert prediction in (0, 1)


def test_threshold_is_valid(threshold):
    assert 0.0 < threshold < 1.0


def test_metadata_matches_threshold(metadata, threshold):
    assert metadata["decision_threshold"] == pytest.approx(threshold)


def test_missing_column_is_rejected(model, valid_input):
    invalid_input = valid_input.drop(columns=["torque_nm"])

    with pytest.raises((ValueError, KeyError)):
        model.predict_proba(invalid_input)


def test_invalid_machine_type_is_handled(model, valid_input):
    # OneHotEncoder(handle_unknown="ignore") should handle
    # an unseen category without crashing.
    invalid_category = valid_input.copy()
    invalid_category["type"] = "UNKNOWN"

    probabilities = model.predict_proba(invalid_category)

    assert probabilities.shape == (1, 2)
    assert np.isfinite(probabilities).all()


def test_missing_numeric_value_is_handled(model, valid_input):
    # The saved pipeline includes median imputation.
    input_with_missing_value = valid_input.copy()
    input_with_missing_value.loc[0, "torque_nm"] = np.nan

    probabilities = model.predict_proba(input_with_missing_value)

    assert probabilities.shape == (1, 2)
    assert np.isfinite(probabilities).all()
