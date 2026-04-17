import os
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

def get_db_engine():
    """Crée un engine SQLAlchemy compatible Postgres + SQLite"""
    database_url = os.getenv('DATABASE_URL')

    if not database_url:
        raise RuntimeError("DATABASE_URL must be set")

    connect_args = {}
    if database_url.startswith("sqlite"):
        connect_args["check_same_thread"] = False

    return create_engine(database_url, connect_args=connect_args)

def is_database_empty(engine):
    """Vérifie si la base de données est vide"""
    with engine.connect() as conn:
        try:
            result = conn.execute(text("SELECT COUNT(*) FROM sirh_employee"))
            count = result.scalar()
            return count == 0
        except Exception:
            return True

# Engine créé une fois
engine = get_db_engine()

# Session factory liée à cet engine
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()