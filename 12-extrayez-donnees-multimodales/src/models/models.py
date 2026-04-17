from sqlalchemy import (
    Column,
    Integer,
    Text,
    Boolean,
    Float,
    DateTime,
    func,
)
from sqlalchemy.orm import declarative_base

Base = declarative_base()


class EtlMetrics(Base):
    __tablename__ = "etl_metrics"

    id = Column(Integer, primary_key=True)

    pipeline_name = Column(Text, nullable=False)
    step = Column(Text, nullable=False)
    run_id = Column(Text, nullable=False)
    execution_date = Column(DateTime)

    # KPI volumétrie
    nb_input = Column(Integer)
    nb_output = Column(Integer)
    nb_rejected = Column(Integer)

    # KPI performance
    duration_seconds = Column(Float)
    rows_per_second = Column(Float)

    # Statut simplifié
    success = Column(Boolean, nullable=False)

    created_at = Column(DateTime, server_default=func.now())
