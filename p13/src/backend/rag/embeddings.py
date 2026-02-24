import json
import os
import time

from sentence_transformers import SentenceTransformer
from pymilvus import (
    connections,
    utility,
    Collection,
    FieldSchema,
    CollectionSchema,
    DataType,
)


DATA_DIR = os.getenv("DATA_DIR", ".data")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")

MILVUS_HOST = os.getenv("MILVUS_HOST", "p13-milvus")
MILVUS_PORT = os.getenv("MILVUS_PORT", "19530")

COLLECTION = "wikichess"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def connect_milvus_grpc(host: str, port: str, retries: int = 30, delay_s: float = 1.0):
    last_err = None
    for _ in range(retries):
        try:
            connections.connect(alias="default", host=host, port=port)
            return
        except Exception as e:
            last_err = e
            time.sleep(delay_s)
    raise RuntimeError(f"Milvus indisponible sur {host}:{port}: {last_err}")


def main():
    chunks_path = os.path.join(PROCESSED_DIR, "wikichess_chunks.jsonl")
    if not os.path.exists(chunks_path):
        raise RuntimeError("Run scrape_wikichess first")

    print("Chargement des chunks...")
    rows = []
    with open(chunks_path, "r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))

    print(f"{len(rows)} chunks chargés")

    print("Embedding...")
    model = SentenceTransformer(MODEL_NAME)

    texts = [r["text"] for r in rows]
    vectors = model.encode(texts, normalize_embeddings=True, show_progress_bar=True)

    print("Connexion Milvus (gRPC)...")
    connect_milvus_grpc(MILVUS_HOST, MILVUS_PORT)

    if not utility.has_collection(COLLECTION):
        print("Création collection...")
        fields = [
            FieldSchema(name="id", dtype=DataType.VARCHAR, is_primary=True, max_length=64),
            FieldSchema(name="url", dtype=DataType.VARCHAR, max_length=512),
            FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=len(vectors[0])),
        ]
        schema = CollectionSchema(fields, description="Wikichess chunks")
        collection = Collection(name=COLLECTION, schema=schema)

        collection.create_index(
            field_name="embedding",
            index_params={
                "index_type": "HNSW",
                "metric_type": "COSINE",
                "params": {"M": 16, "efConstruction": 200},
            },
        )
    else:
        collection = Collection(COLLECTION)

    print("Insertion dans Milvus...")
    ids = [r["id"] for r in rows]
    urls = [r["url"] for r in rows]
    texts = [r["text"] for r in rows]
    embeddings = [v.tolist() for v in vectors]

    collection.insert([ids, urls, texts, embeddings])
    collection.load()  # utile pour la recherche

    print("Indexation terminée.")


if __name__ == "__main__":
    main()