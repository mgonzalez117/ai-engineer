from __future__ import annotations

from fastapi import APIRouter, HTTPException

from backend.service.lichess.lichess import LichessService, LichessError
from backend.service.utils import validate_fen_or_raise

router = APIRouter(prefix="/api/v1", tags=["chess"])


@router.get(
    "/moves/{fen:path}",
    summary="Récupère les coups théoriques depuis Lichess",
    description=(
        "Retourne les coups théoriques connus pour une position donnée au format FEN.\n\n"
        "Exemple de FEN (position initiale) :\n"
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1\n\n"
        "Exemple d’appel (URL encodée) :\n"
        "GET /api/v1/moves/"
        "rnbqkbnr%2Fpppppppp%2F8%2F8%2F8%2F8%2FPPPPPPPP%2FRNBQKBNR%20w%20KQkq%20-%200%201"
    ),
    responses={
        200: {
            "description": "Liste des coups théoriques",
            "content": {
                "application/json": {
                    "example": {
                        "fen": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
                        "moves": [
                            {"uci": "e2e4", "san": "e4"},
                            {"uci": "d2d4", "san": "d4"},
                            {"uci": "c2c4", "san": "c4"}
                        ]
                    }
                }
            },
        },
        400: {"description": "FEN invalide"},
        502: {"description": "Erreur lors de l'appel à l'API Lichess"},
    },
)
async def get_moves(fen: str):
    """
    Endpoint permettant de récupérer les coups théoriques d'une position FEN.
    """
    try:
        validate_fen_or_raise(fen)
    except ValueError:
        raise HTTPException(status_code=400, detail="FEN invalide")

    service = LichessService()
    try:
        moves = await service.get_theoretical_moves(fen)
    except LichessError as e:
        raise HTTPException(status_code=502, detail=str(e))

    return {"fen": fen, "moves": moves}