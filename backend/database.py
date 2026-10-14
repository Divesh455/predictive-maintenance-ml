from pathlib import Path

from sqlalchemy import (
    Column,
    DateTime,
    Float,
    Integer,
    String,
    create_engine,
)
from sqlalchemy.orm import declarative_base, sessionmaker
from sqlalchemy.sql import func

import os

ROOT = Path(__file__).resolve().parent.parent
DATABASE_DIR = ROOT / "data"
DATABASE_DIR.mkdir(parents=True, exist_ok=True)

# Prefer DATABASE_URL from environment (Render injects Postgres URL if set),
# otherwise use local SQLite file for development.
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    f"sqlite:///{(DATABASE_DIR / 'predictions.db').as_posix()}",
)

# SQLAlchemy < 2 still used "postgres://" prefix — normalize to "postgresql://"
if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

_connect_args = {"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}

engine = create_engine(
    DATABASE_URL,
    connect_args=_connect_args,
)

SessionLocal = sessionmaker(
    bind=engine,
    autoflush=False,
    autocommit=False,
)

Base = declarative_base()


class PredictionHistory(Base):
    __tablename__ = "prediction_history"

    id = Column(Integer, primary_key=True, index=True)
    created_at = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )

    machine_type = Column(String(1), nullable=False)
    air_temperature_k = Column(Float, nullable=False)
    process_temperature_k = Column(Float, nullable=False)
    rotational_speed_rpm = Column(Integer, nullable=False)
    torque_nm = Column(Float, nullable=False)
    tool_wear_min = Column(Integer, nullable=False)

    model_name = Column(String(100), nullable=False)
    failure_probability = Column(Float, nullable=False)
    decision_threshold = Column(Float, nullable=False)
    predicted_failure = Column(Integer, nullable=False)
    status = Column(String(50), nullable=False)


def init_db():
    Base.metadata.create_all(bind=engine)
