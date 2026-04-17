import os
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from src.data.models import PredictLogs

def get_db_engine():
    """Crée un engine SQLAlchemy compatible Postgres + SQLite"""
    database_url = os.getenv('DATABASE_URL')

    if not database_url:
        raise RuntimeError("DATABASE_URL must be set")

    connect_args = {}
    if database_url.startswith("sqlite"):
        connect_args["check_same_thread"] = False

    return create_engine(database_url, connect_args=connect_args)

# Engine créé une fois
engine = get_db_engine()

# Création des tables si elles n'existent pas
PredictLogs.metadata.create_all(bind=engine)

# Session factory liée à cet engine
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()