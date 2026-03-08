from __future__ import annotations

from typing import Any, Literal, Optional, TypedDict


class AgentState(TypedDict, total=False):
    fen: str

    # Résultats
    theory_moves: list[dict[str, Any]]
    evaluation: dict[str, Any]
    context: list[dict[str, Any]]
    videos: list[dict[str, Any]]

    # Métadonnées
    source: Optional[Literal["lichess", "stockfish"]]
    warnings: list[str]
    error: Optional[str]

    # Sortie normalisée pour le front
    final: dict[str, Any]