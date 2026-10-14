
import json
import logging
from pathlib import Path
from typing import Literal

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import desc
from sqlalchemy.exc import SQLAlchemyError

from app.database import (
    PredictionHistory,
    SessionLocal,
    init_db,
)


# ============================================================
# 1. Configuration
# ============================================================

ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = ROOT / "models" / "final_model.joblib"
THRESHOLD_PATH = ROOT / "reports" / "final_decision_threshold.json"
METRICS_PATH = ROOT / "reports" / "final_test_metrics.json"
METADATA_PATH = ROOT / "reports" / "final_model_metadata.json"

FEATURES = [
    "type",
    "air_temperature_k",
    "process_temperature_k",
    "rotational_speed_rpm",
    "torque_nm",
    "tool_wear_min",
]

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================
# 2. FastAPI application
# ============================================================

app = FastAPI(
    title="Predictive Maintenance API",
    description=(
        "Predict machine failure risk using the final trained "
        "machine learning pipeline and save prediction history."
    ),
    version="1.0.0",
)


# ============================================================
# 3. Load saved model and configuration
# ============================================================

try:
    model = joblib.load(MODEL_PATH)

    with open(THRESHOLD_PATH, encoding="utf-8") as file:
        threshold_config = json.load(file)

    with open(METADATA_PATH, encoding="utf-8") as file:
        metadata = json.load(file)

    with open(METRICS_PATH, encoding="utf-8") as file:
        test_metrics = json.load(file)

    threshold = float(threshold_config["threshold"])

    if not 0.0 < threshold < 1.0:
        raise ValueError("Decision threshold must be between 0 and 1.")

    if list(metadata["predictor_columns"]) != FEATURES:
        raise ValueError(
            "Model metadata features do not match the API feature order."
        )

    if metadata["model_name"] != test_metrics["selected_model"]:
        raise ValueError(
            "Model metadata and test metrics have different model names."
        )

except Exception as exc:
    raise RuntimeError(
        f"Could not load model artifacts: {exc}"
    ) from exc


# ============================================================
# 4. Initialize database
# ============================================================

try:
    init_db()
    logger.info("Prediction history database initialized.")
except Exception as exc:
    logger.exception("Database initialization failed.")
    raise RuntimeError(
        "Could not initialize the prediction history database."
    ) from exc


# ============================================================
# 5. Request and response schemas
# ============================================================

class MachineInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["L", "M", "H"] = Field(
        description="Product quality variant: L, M, or H."
    )

    air_temperature_k: float = Field(
        ge=250,
        le=350,
        allow_inf_nan=False,
        description="Air temperature in Kelvin.",
    )

    process_temperature_k: float = Field(
        ge=250,
        le=400,
        allow_inf_nan=False,
        description="Process temperature in Kelvin.",
    )

    rotational_speed_rpm: int = Field(
        gt=0,
        le=10000,
        description="Rotational speed in revolutions per minute.",
    )

    torque_nm: float = Field(
        ge=0,
        le=500,
        allow_inf_nan=False,
        description="Torque in Newton-metres.",
    )

    tool_wear_min: int = Field(
        ge=0,
        le=1000,
        description="Tool wear in minutes.",
    )


class PredictionResponse(BaseModel):
    model_name: str
    failure_probability: float
    decision_threshold: float
    predicted_failure: int
    status: str
    recommendation: str


# ============================================================
# 6. Root endpoint
# ============================================================

@app.get("/")
def root():
    return {
        "message": "Predictive Maintenance API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
        "predict": "/predict",
        "history": "/history",
        "model_info": "/model-info",
    }


# ============================================================
# 7. Health check
# ============================================================

@app.get("/health")
def health():
    db = SessionLocal()

    try:
        # Check database connectivity.
        from sqlalchemy import text

        db.execute(text("SELECT 1"))

        return {
            "status": "healthy",
            "model_loaded": model is not None,
            "database_connected": True,
        }

    except SQLAlchemyError as exc:
        logger.exception("Health check database failure.")

        raise HTTPException(
            status_code=503,
            detail="Database is unavailable.",
        ) from exc

    finally:
        db.close()


# ============================================================
# 8. Model information
# ============================================================

@app.get("/model-info")
def model_info():
    return {
        "model_name": metadata["model_name"],
        "target": metadata["target"],
        "predictor_columns": metadata["predictor_columns"],
        "decision_threshold": threshold,
        "selection_metric": metadata["selection_metric"],
        "threshold_selection_metric": metadata[
            "threshold_selection_metric"
        ],
        "test_metrics": test_metrics,
    }


# ============================================================
# 9. Prediction endpoint
# ============================================================

@app.post("/predict", response_model=PredictionResponse)
def predict(data: MachineInput):
    row = pd.DataFrame(
        [data.model_dump()],
        columns=FEATURES,
    )

    try:
        probabilities = model.predict_proba(row)
        probability = float(probabilities[0, 1])

        prediction = int(probability >= threshold)

    except Exception as exc:
        logger.exception("Model prediction failed.")

        raise HTTPException(
            status_code=500,
            detail="Prediction failed. Check server logs.",
        ) from exc

    if prediction == 1:
        status = "FAILURE_RISK"
        recommendation = (
            "The model flagged this machine for failure risk. "
            "Arrange an appropriate inspection and review the "
            "machine measurements. This is not a confirmed failure."
        )
    else:
        status = "NORMAL_RISK"
        recommendation = (
            "The model did not flag this machine at the configured "
            "threshold. Continue routine monitoring."
        )

    # Save the prediction and measurements in SQLite.
    db = SessionLocal()

    try:
        history = PredictionHistory(
            machine_type=data.type,
            air_temperature_k=data.air_temperature_k,
            process_temperature_k=data.process_temperature_k,
            rotational_speed_rpm=data.rotational_speed_rpm,
            torque_nm=data.torque_nm,
            tool_wear_min=data.tool_wear_min,
            model_name=metadata["model_name"],
            failure_probability=probability,
            decision_threshold=threshold,
            predicted_failure=prediction,
            status=status,
        )

        db.add(history)
        db.commit()

    except SQLAlchemyError as exc:
        db.rollback()
        logger.exception("Could not save prediction history.")

        raise HTTPException(
            status_code=500,
            detail="Prediction was calculated, but history could not be saved.",
        ) from exc

    finally:
        db.close()

    return PredictionResponse(
        model_name=metadata["model_name"],
        failure_probability=round(probability, 6),
        decision_threshold=threshold,
        predicted_failure=prediction,
        status=status,
        recommendation=recommendation,
    )


# ============================================================
# 10. Prediction history endpoint
# ============================================================

@app.get("/history")
def get_prediction_history(
    limit: int = Query(default=50, ge=1, le=500),
):
    db = SessionLocal()

    try:
        rows = (
            db.query(PredictionHistory)
            .order_by(desc(PredictionHistory.id))
            .limit(limit)
            .all()
        )

        return [
            {
                "id": row.id,
                "created_at": (
                    row.created_at.isoformat()
                    if row.created_at
                    else None
                ),
                "machine_type": row.machine_type,
                "air_temperature_k": row.air_temperature_k,
                "process_temperature_k": row.process_temperature_k,
                "rotational_speed_rpm": row.rotational_speed_rpm,
                "torque_nm": row.torque_nm,
                "tool_wear_min": row.tool_wear_min,
                "model_name": row.model_name,
                "failure_probability": row.failure_probability,
                "decision_threshold": row.decision_threshold,
                "predicted_failure": row.predicted_failure,
                "status": row.status,
            }
            for row in rows
        ]

    except SQLAlchemyError as exc:
        logger.exception("Could not retrieve prediction history.")

        raise HTTPException(
            status_code=500,
            detail="Could not retrieve prediction history.",
        ) from exc

    finally:
        db.close()
