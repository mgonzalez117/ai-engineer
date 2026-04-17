# models/drift_feature_metric.py
from sqlalchemy import Column, Integer, String, Float, Boolean, ForeignKey
from sqlalchemy.orm import relationship
from src.data.models.base import Base

class DriftFeatureMetric(Base):
    """
    Détaille les métriques de drift pour chaque feature au sein d'un run spécifique.
    """
    __tablename__ = "drift_feature_metric"

    id = Column(Integer, primary_key=True, autoincrement=True)

    # Clé étrangère vers drift_run
    run_id = Column(Integer, ForeignKey("drift_run.id"), nullable=False, index=True)

    feature_name = Column(String(100), nullable=False, index=True)

    drift_detected = Column(Boolean, nullable=False)
    drift_score = Column(Float, nullable=True)
    stattest_name = Column(String(50), nullable=True) # type de test statistique

    # Relation pour accéder au run parent depuis la métrique de feature
    drift_run = relationship("DriftRun", backref="feature_metrics")

    def __repr__(self):
        return (
            f"<DriftFeatureMetric(id={self.id}, run_id={self.run_id}, "
            f"feature={self.feature_name}, drift={self.drift_detected})>"
        )