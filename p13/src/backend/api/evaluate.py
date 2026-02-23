from __future__ import annotations

from fastapi import APIRouter, HTTPException

from backend.service.stockfish.stockfish import StockfishService, StockfishError
from backend.service.utils import validate_fen_or_raise

router = APIRouter(prefix="/api/v1", tags=["chess"])


@router.get("/evaluate/{fen:path}")
def evaluate(fen: str):
    try:
        validate_fen_or_raise(fen)
    except ValueError:
        raise HTTPException(status_code=400, detail="FEN invalide")

    try:
        sf = StockfishService()
        evaluation = sf.evaluate_fen(fen)
    except StockfishError as e:
        raise HTTPException(status_code=500, detail=str(e))

    return {"fen": fen, "evaluation": evaluation}