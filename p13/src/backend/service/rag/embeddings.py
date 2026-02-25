import os
from typing import List, Sequence

from sentence_transformers import SentenceTransformer

MODEL_NAME = os.getenv("EMBEDDING_MODEL")

_model: SentenceTransformer | None = None


def get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(MODEL_NAME)
    return _model


def embed_text(text: str) -> List[float]:
    """
    Embedding normalisé d'un texte (usage runtime: vector search).
    """
    vec = get_model().encode(text, normalize_embeddings=True)
    return vec.tolist()


def embed_texts(texts: Sequence[str], show_progress_bar: bool = False) -> List[List[float]]:
    """
    Embedding normalisé d'une liste de textes (usage ingest).
    Retourne une liste de listes (compatible JSON/Milvus).
    """
    vectors = get_model().encode(
        list(texts),
        normalize_embeddings=True,
        show_progress_bar=show_progress_bar,
    )
    return [v.tolist() for v in vectors]