import json
import logging
from pathlib import Path
from typing import Literal

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import desc
from sqlalchemy.exc import SQLAlchemyError

from backend.database import (
    PredictionHistory,
    SessionLocal,
    init_db,
)

# ============================================================
# 1. Configuration & Paths
# ============================================================

ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = ROOT / "models" / "final_model.joblib"
THRESHOLD_PATH = ROOT / "reports" / "final_decision_threshold.json"
METRICS_PATH = ROOT / "reports" / "final_test_metrics.json"
METADATA_PATH = ROOT / "reports" / "final_model_metadata.json"
SHAP_PATH = ROOT / "reports" / "final_shap_feature_importance.csv"

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
# 2. FastAPI Application & CORS
# ============================================================

app = FastAPI(
    title="Predictive Maintenance API",
    description=(
        "Production ML Inference & Monitoring API for industrial equipment failure risk detection."
    ),
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================
# 3. Load Model Artifacts
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
        raise ValueError("Model metadata features do not match API feature order.")

    logger.info("Machine learning model and metadata loaded successfully.")

except Exception as exc:
    logger.exception("Could not load model artifacts.")
    raise RuntimeError(f"Could not load model artifacts: {exc}") from exc

# ============================================================
# 4. Initialize Database
# ============================================================

try:
    init_db()
    logger.info("Prediction history database initialized.")
except Exception as exc:
    logger.exception("Database initialization failed.")
    raise RuntimeError("Could not initialize the database.") from exc

# ============================================================
# 5. Request & Response Schemas
# ============================================================

class MachineInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["L", "M", "H"] = Field(
        description="Product quality variant: L (50%), M (30%), or H (20%)."
    )
    air_temperature_k: float = Field(
        ge=250.0,
        le=350.0,
        allow_inf_nan=False,
        description="Ambient air temperature in Kelvin (e.g. 300.0).",
    )
    process_temperature_k: float = Field(
        ge=250.0,
        le=400.0,
        allow_inf_nan=False,
        description="Process operational temperature in Kelvin (e.g. 310.0).",
    )
    rotational_speed_rpm: int = Field(
        gt=0,
        le=10000,
        description="Spindle rotational speed in RPM (e.g. 1500).",
    )
    torque_nm: float = Field(
        ge=0.0,
        le=500.0,
        allow_inf_nan=False,
        description="Applied rotational torque in Newton-metres (e.g. 40.0).",
    )
    tool_wear_min: int = Field(
        ge=0,
        le=1000,
        description="Accumulated cutting tool contact time in minutes (e.g. 120).",
    )


class PredictionResponse(BaseModel):
    model_name: str
    failure_probability: float
    decision_threshold: float
    predicted_failure: int
    status: str
    recommendation: str
    risk_level: str


# ============================================================
# 6. Endpoints
# ============================================================

@app.get("/")
def root():
    return {
        "message": "Predictive Maintenance API",
        "version": "2.0.0",
        "status": "online",
        "endpoints": {
            "health": "/health",
            "predict": "/predict",
            "history": "/history",
            "model_info": "/model-info",
            "feature_importance": "/feature-importance",
            "docs": "/docs",
        },
    }


@app.get("/health")
def health():
    db = SessionLocal()
    try:
        from sqlalchemy import text
        db.execute(text("SELECT 1"))
        return {
            "status": "healthy",
            "model_loaded": model is not None,
            "database_connected": True,
            "selected_model": metadata.get("model_name", "Unknown"),
            "threshold": threshold,
        }
    except SQLAlchemyError as exc:
        logger.exception("Health check database failure.")
        raise HTTPException(status_code=503, detail="Database unavailable.") from exc
    finally:
        db.close()


@app.get("/model-info")
def model_info():
    return {
        "model_name": metadata["model_name"],
        "target": metadata["target"],
        "predictor_columns": metadata["predictor_columns"],
        "decision_threshold": threshold,
        "selection_metric": metadata.get("selection_metric"),
        "threshold_selection_metric": metadata.get("threshold_selection_metric"),
        "test_metrics": test_metrics,
    }


@app.get("/feature-importance")
def feature_importance():
    if SHAP_PATH.exists():
        df = pd.read_csv(SHAP_PATH)
        return df.to_dict(orient="records")
    return []


@app.post("/predict", response_model=PredictionResponse)
def predict(data: MachineInput):
    row = pd.DataFrame([data.model_dump()], columns=FEATURES)

    try:
        probabilities = model.predict_proba(row)
        probability = float(probabilities[0, 1])
        prediction = int(probability >= threshold)
    except Exception as exc:
        logger.exception("Model prediction inference failed.")
        raise HTTPException(
            status_code=500, detail="Inference computation failed."
        ) from exc

    # Determine risk category
    if probability >= threshold:
        status = "FAILURE_RISK"
        risk_level = "Critical"
        recommendation = (
            "CRITICAL: High failure probability detected exceeding decision threshold ("
            f"{threshold:.0%}). Schedule immediate diagnostic stoppage, verify tool wear, "
            "and inspect spindle torque loads."
        )
    elif probability >= 0.40:
        status = "WARNING_RISK"
        risk_level = "Elevated"
        recommendation = (
            "WARNING: Elevated failure probability observed. Operating condition is approaching "
            "critical margin. Plan tool inspection and monitor motor temperature."
        )
    else:
        status = "NORMAL_RISK"
        risk_level = "Nominal"
        recommendation = (
            "NOMINAL: Machine operation is within safe parameters. Continue standard preventative maintenance cycle."
        )

    # Persist in SQLite
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
        logger.exception("Failed to write prediction history.")
        raise HTTPException(
            status_code=500, detail="Prediction succeeded but history persistence failed."
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
        risk_level=risk_level,
    )


@app.get("/history")
def get_prediction_history(limit: int = Query(default=50, ge=1, le=500)):
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
                "created_at": row.created_at.isoformat() if row.created_at else None,
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
        logger.exception("Failed to fetch history.")
        raise HTTPException(status_code=500, detail="Could not retrieve history.") from exc
    finally:
        db.close()
