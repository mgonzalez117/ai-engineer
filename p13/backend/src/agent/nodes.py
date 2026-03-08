from __future__ import annotations

from typing import Any

import chess

from agent.state import AgentState
from service.lichess.lichess import LichessService, LichessError
from service.stockfish.stockfish import StockfishService, StockfishError
from service.rag.vector_search import search_chunks
from service.youtube.youtube import YouTubeService, YouTubeServiceError


def _add_warning(state: AgentState, msg: str) -> AgentState:
    warnings = list(state.get("warnings", []))
    warnings.append(msg)
    return {"warnings": warnings}


def validate_fen_node(state: AgentState) -> AgentState:
    fen = state["fen"]
    try:
        chess.Board(fen)
    except Exception as e:
        return {"error": f"FEN invalide: {e}"}
    return {}


async def get_theoretical_moves_node(state: AgentState) -> AgentState:
    if state.get("error"):
        return {}

    fen = state["fen"]

    try:
        lichess = LichessService()
        moves: list[dict[str, Any]] = await lichess.get_theoretical_moves(fen)
    except LichessError as e:
        patch = _add_warning(state, f"Lichess indisponible: {e}")
        patch.update({"theory_moves": []})
        return patch

    if moves:
        return {"theory_moves": moves, "source": "lichess"}

    return {"theory_moves": []}


def evaluate_stockfish_node(state: AgentState) -> AgentState:
    if state.get("error"):
        return {}

    fen = state["fen"]

    try:
        sf = StockfishService()
        evaluation: dict[str, Any] = sf.evaluate_fen(fen)
    except StockfishError as e:
        return {"error": f"Stockfish error: {e}"}

    return {"evaluation": evaluation, "source": "stockfish"}


def search_milvus_node(state: AgentState) -> AgentState:
    if state.get("error"):
        return {}

    # Minimal et utile :
    # - si on a des coups théoriques, on interroge sur "chess opening"
    # - sinon idem, on reste générique
    # Plus tard on pourra améliorer avec un vrai nom d'ouverture.
    query = "chess opening"

    try:
        context = search_chunks(query=query, top_k=5)
    except Exception as e:
        return _add_warning(state, f"Milvus indisponible: {e}")

    return {"context": context or []}


def search_youtube_node(state: AgentState) -> AgentState:
    if state.get("error"):
        return {}

    # Même logique minimale : requête générique stable
    opening = "chess opening"

    try:
        youtube = YouTubeService()
        videos = youtube.search_videos(opening=opening, max_results=5)
    except YouTubeServiceError as e:
        return _add_warning(state, f"YouTube indisponible: {e}")
    except Exception as e:
        return _add_warning(state, f"YouTube indisponible: {e}")

    return {"videos": videos or []}


def compose_response_node(state: AgentState) -> AgentState:
    if state.get("error"):
        return {
            "final": {
                "fen": state.get("fen"),
                "source": None,
                "recommendations": [],
                "evaluation": None,
                "context": [],
                "videos": [],
                "warnings": state.get("warnings", []),
                "error": state["error"],
            }
        }

    final = {
        "fen": state["fen"],
        "source": state.get("source"),
        "recommendations": state.get("theory_moves", []),
        "evaluation": state.get("evaluation"),
        "context": state.get("context", []),
        "videos": state.get("videos", []),
        "warnings": state.get("warnings", []),
        "error": None,
    }
    return {"final": final}