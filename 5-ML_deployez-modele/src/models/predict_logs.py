from sqlalchemy import Column, Integer, String, DateTime, JSON, Float, Text
from datetime import datetime
from src.models.base import Base

class PredictLogs(Base):
    __tablename__ = "predict_logs"

    id = Column(Integer, primary_key=True, index=True)
    date = Column(DateTime, default=datetime.utcnow, nullable=False)

    # Input data
    input_payload = Column(JSON, nullable=False)  # Les données envoyées au modèle

    # Output data
    prediction_result = Column(JSON, nullable=True)  # Le résultat de la prédiction

    # Metadata
    processing_time_ms = Column(Float, nullable=True)  # Temps de traitement en ms
    status = Column(String(20), default="success")  # success, error, timeout
    error_message = Column(Text, nullable=True)  # Message d'erreur si applicable