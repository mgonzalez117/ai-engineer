import os
from typing import Any, Dict, List

from pymilvus import connections, Collection

from backend.service.rag.embeddings import embed_text

MILVUS_HOST = os.getenv("MILVUS_HOST", "p13-milvus")
MILVUS_PORT = os.getenv("MILVUS_PORT", "19530")
COLLECTION_NAME = os.getenv("MILVUS_COLLECTION", "wikichess")

def _connect():
    """
    Connexion gRPC simple à Milvus
    """
    connections.connect(alias="default", host=MILVUS_HOST, port=MILVUS_PORT)


def search_chunks(query: str, top_k: int = 5) -> List[Dict[str, Any]]:
    """
    Recherche vectorielle simple dans Milvus.
    """
    _connect()

    collection = Collection(COLLECTION_NAME)
    collection.load()

    vector = embed_text(query)

    results = collection.search(
        data=[vector],
        anns_field="embedding",
        param={
            "metric_type": "COSINE",
            "params": {"ef": 128},  # standard pour HNSW Hierarchical Navigable Small World (algorithme d'indexation)
        },
        limit=top_k,
        output_fields=["text", "url"],
    )

    hits: List[Dict[str, Any]] = []

    for hit in results[0]:
        hits.append(
            {
                "id": hit.id,
                "score": float(hit.score),
                "text": hit.entity.get("text"),
                "url": hit.entity.get("url"),
            }
        )

    return hits