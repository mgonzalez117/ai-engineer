from fastapi import APIRouter, Depends, HTTPException
from fastapi.security import HTTPBearer
from src.data.build import build_index

router = APIRouter()
security = HTTPBearer()

@router.put("/rebuild", dependencies=[Depends(security)])
async def rebuild_index():
    """Reconstruit l'index FAISS depuis l'API"""
    result = build_index()

    if result['success']:
        return {
            "status": "success",
            "message": result['message'],
            "details": {
                "num_events": result['num_events'],
                "num_chunks": result['num_chunks'],
                "avg_chunks_per_event": result['avg_chunks_per_event'],
                "embedding_model": result['embedding_model'],
                "dimension": result['dimension']
            }
        }
    else:
        raise HTTPException(
            status_code=500,
            detail={
                "status": "error",
                "message": result['message'],
                "num_events": result['num_events'],
                "num_chunks": result['num_chunks']
            }
        )