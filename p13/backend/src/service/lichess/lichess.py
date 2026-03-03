from __future__ import annotations

import os
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
    base_url: str = os.getenv("LICHESS_EXPLORER_BASE_URL", "https://explorer.lichess.ovh")
    timeout_s: float = float(os.getenv("LICHESS_TIMEOUT_S", "5.0"))

    async def get_theoretical_moves(self, fen: str, variant: str = "standard") -> list[dict[str, Any]]:
        url = f"{self.base_url}/masters"
        params = {"fen": fen, "variant": variant}

        try:
            async with httpx.AsyncClient(timeout=self.timeout_s, trust_env=False) as client:
                r = await client.get(url, params=params)
        except httpx.TimeoutException as e:
            raise LichessError(f"Lichess timeout: {e}") from e
        except httpx.RequestError as e:
            raise LichessError(f"Lichess request error: {e}") from e

        if r.status_code == 429:
            raise LichessError("Lichess rate limit (429).")

        if r.status_code >= 400:
            # évite de spammer des pages HTML entières dans le detail
            body_head = (r.text or "")[:300].replace("\n", " ").replace("\r", " ")
            raise LichessError(f"Lichess error {r.status_code} on {url}: {body_head}")

        try:
            data = r.json()
        except ValueError as e:
            body_head = (r.text or "")[:300].replace("\n", " ").replace("\r", " ")
            raise LichessError(f"Lichess returned non-JSON response: {body_head}") from e

        return data.get("moves", [])