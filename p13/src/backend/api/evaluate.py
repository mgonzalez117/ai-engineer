from __future__ import annotations

from fastapi import APIRouter, HTTPException

from backend.service.stockfish.stockfish import StockfishService, StockfishError
from backend.service.utils import validate_fen_or_raise

router = APIRouter(prefix="/api/v1", tags=["chess"])


@router.get(
    "/evaluate/{fen:path}",
    summary="Évalue une position avec Stockfish",
    description=(
        "Retourne l’évaluation d’une position donnée au format FEN via le moteur Stockfish.\n\n"
        "L’évaluation est généralement exprimée en centipawns (cp).\n"
        "Une valeur positive indique un avantage pour les blancs.\n"
        "Une valeur négative indique un avantage pour les noirs.\n\n"
        "Exemple de FEN après 1.e4 :\n"
        "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1\n\n"
        "Exemple d’appel (URL encodée) :\n"
        "GET /api/v1/evaluate/"
        "rnbqkbnr%2Fpppppppp%2F8%2F8%2F4P3%2F8%2FPPPP1PPP%2FRNBQKBNR%20b%20KQkq%20-%200%201"
    ),
    responses={
        200: {
            "description": "Évaluation retournée par Stockfish",
            "content": {
                "application/json": {
                    "example": {
                        "fen": "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1",
                        "evaluation": {
                            "type": "cp",
                            "value": 20
                        }
                    }
                }
            },
        },
        400: {"description": "FEN invalide"},
        500: {"description": "Erreur interne Stockfish"},
    },
)
def evaluate(fen: str):
    """
    Endpoint permettant d’évaluer une position FEN avec Stockfish.
    """
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