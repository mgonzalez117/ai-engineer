import json
import os

from sqlalchemy import create_engine, MetaData, Table
from sqlalchemy.dialects.postgresql import insert

PUBLICATION_COLUMNS = {
    "id",
    "source",
    "title",
    "fake",
    "content",
    "image_url",
    "document_url",
    "video_url",
    "author",
}


def load_publications_jsonl(input_path: str):
    """
    Generic loader: JSONL (neutral transformed format) -> publication table
    Uses AIRFLOW__DATABASE__SQL_ALCHEMY_CONN
    """

    engine_url = os.getenv("AIRFLOW__DATABASE__SQL_ALCHEMY_CONN")
    engine = create_engine(engine_url)

    metadata = MetaData()
    publication = Table("publication", metadata, autoload_with=engine)

    with engine.begin() as conn:
        with open(input_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue

                obj = json.loads(line)

                # keep only valid columns
                row = {k: obj.get(k) for k in PUBLICATION_COLUMNS if k in obj}

                # required fields
                for k in ("source", "id", "title", "fake", "image_url"):
                    if row.get(k) in (None, ""):
                        raise ValueError(f"Missing required field: {k}")

                stmt = insert(publication).values(**row)

                # upsert (PostgreSQL)
                update_cols = {
                    c.name: stmt.excluded[c.name]
                    for c in publication.columns
                    if c.name not in ("source", "id")
                }

                stmt = stmt.on_conflict_do_update(
                    index_elements=["source", "id"],
                    set_=update_cols,
                )

                conn.execute(stmt)
