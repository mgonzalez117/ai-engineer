from __future__ import annotations

from fastapi import APIRouter, HTTPException

from backend.service.lichess.lichess import LichessService, LichessError
from backend.service.utils import validate_fen_or_raise

router = APIRouter(prefix="/api/v1", tags=["chess"])


@router.get("/moves/{fen:path}")
async def get_moves(fen: str):
    try:
        validate_fen_or_raise(fen)
    except ValueError:
        raise HTTPException(status_code=400, detail="FEN invalide")

    service = LichessService()
    try:
        moves = await service.get_theoretical_moves(fen)
    except LichessError as e:
        raise HTTPException(status_code=502, detail=str(e))

    # Tu peux renvoyer brut ou normalisé; ici brut minimal
    return {"fen": fen, "moves": moves}