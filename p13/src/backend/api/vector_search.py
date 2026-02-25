from fastapi import APIRouter
from pydantic import BaseModel, Field

from backend.service.rag.vector_search import search_chunks

router = APIRouter(prefix="/api/v1", tags=["chess"])

class VectorSearchRequest(BaseModel):
    query: str = Field(..., min_length=2)
    top_k: int = Field(5, ge=1, le=20)

@router.post("/vector-search")
async def vector_search(payload: VectorSearchRequest):
    """
    Recherche vectorielle dans la base Wikichess.
    """
    results = search_chunks(payload.query, top_k=payload.top_k)

    return {
        "query": payload.query,
        "top_k": payload.top_k,
        "results": results,
    }