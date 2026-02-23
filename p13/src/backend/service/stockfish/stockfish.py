# src/backend/service/stockfish/stockfish.py
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

from stockfish import Stockfish


class StockfishError(RuntimeError):
    pass


@dataclass
class StockfishService:
    """
    Wrapper simple autour du package 'stockfish' (python).
    """
    engine_path: str | None = None
    depth: int = 15

    def __post_init__(self) -> None:
        path = self.engine_path or os.getenv("STOCKFISH_PATH")
        if not path:
            raise StockfishError("STOCKFISH_PATH manquant (chemin du binaire stockfish).")

        self._sf = Stockfish(path=path, depth=self.depth)

    def evaluate_fen(self, fen: str) -> dict[str, Any]:
        """
        Retourne un dict type:
        - {"type": "cp", "value": 34}  (centipawns)
        - {"type": "mate", "value": -3} (mate en N)
        Le package stockfish expose get_evaluation().
        """
        self._sf.set_fen_position(fen)
        return self._sf.get_evaluation()