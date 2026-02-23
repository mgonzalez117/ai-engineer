# src/backend/service/lichess/lichess.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import httpx


class LichessError(RuntimeError):
    pass


@dataclass
class LichessService:
    """
    Service d'accès à l'Opening Explorer Lichess.
    """
    base_url: str = "https://explorer.lichess.ovh"
    timeout_s: float = 5.0

    async def get_theoretical_moves(self, fen: str, variant: str = "standard") -> list[dict[str, Any]]:
        """
        Retourne la liste brute des coups depuis l'Opening Explorer.
        Structure typique: {"moves":[{"uci":"e2e4","san":"e4","white":..,"draws":..,"black":..}, ...], ...}
        (Le shape exact dépend de l'endpoint.)
        """
        url = f"{self.base_url}/masters"
        params = {"fen": fen, "variant": variant}

        async with httpx.AsyncClient(timeout=self.timeout_s) as client:
            r = await client.get(url, params=params)

        if r.status_code == 429:
            raise LichessError("Rate limit Lichess (429).")
        if r.status_code >= 400:
            raise LichessError(f"Lichess error {r.status_code}: {r.text}")

        data = r.json()
        return data.get("moves", [])