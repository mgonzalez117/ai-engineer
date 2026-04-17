# models/drift_run.py
from datetime import datetime
from sqlalchemy import Column, Integer, String, Boolean, DateTime, Float
from src.data.models.base import Base

class DriftRun(Base):
    """
    Représente un run de monitoring de drift global (dataset-level).
    """
    __tablename__ = "drift_run"

    id = Column(Integer, primary_key=True, autoincrement=True)

    # Timestamp du calcul
    date = Column(DateTime, nullable=False, default=datetime.utcnow, index=True)

    # Indique si un drift global a été détecté pour le dataset
    dataset_drift = Column(Boolean, nullable=False)

    # Score de drift global (share de colonnes ayant drifté)
    drift_score = Column(Float, nullable=True)

    def __repr__(self):
        return (
            f"<DriftRun(id={self.id}, date={self.date}, "
            f"dataset_drift={self.dataset_drift}, drift_score={self.drift_score})>"
        )