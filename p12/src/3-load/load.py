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

REQUIRED = ("source", "id", "title", "fake", "image_url")


def load_publications_jsonl(input_path: str, batch_size: int = 5000):
    """
    Returns: (nb_input, nb_output, nb_rejected)
    """
    engine_url = os.getenv("AIRFLOW__DATABASE__SQL_ALCHEMY_CONN")
    if not engine_url:
        raise RuntimeError("Missing env var AIRFLOW__DATABASE__SQL_ALCHEMY_CONN")

    engine = create_engine(engine_url)

    metadata = MetaData()
    publication = Table("publication", metadata, autoload_with=engine)

    # Prepared upsert statement (executemany via conn.execute(stmt, list_of_dicts))
    stmt = insert(publication).on_conflict_do_update(
        index_elements=["source", "id"],
        set_={
            c.name: insert(publication).excluded[c.name]
            for c in publication.columns
            if c.name not in ("source", "id")
        },
    )

    nb_input = 0
    nb_output = 0
    nb_rejected = 0

    batch = []

    def flush(conn):
        nonlocal nb_output, batch
        if not batch:
            return
        conn.execute(stmt, batch)  # executemany
        nb_output += len(batch)
        batch = []

    with engine.begin() as conn:
        with open(input_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue

                nb_input += 1
                try:
                    obj = json.loads(line)

                    row = {k: obj.get(k) for k in PUBLICATION_COLUMNS if k in obj}

                    for k in REQUIRED:
                        if row.get(k) in (None, ""):
                            raise ValueError(f"Missing required field: {k}")

                    batch.append(row)

                    if len(batch) >= batch_size:
                        flush(conn)

                except Exception:
                    nb_rejected += 1

            flush(conn)

    return nb_input, nb_output, nb_rejected
