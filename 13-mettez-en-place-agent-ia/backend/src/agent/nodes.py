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

def _build_query_from_fen(fen: str) -> str:
    board = chess.Board(fen)

    # Position initiale exacte
    if fen == chess.STARTING_FEN:
        return "starting position chess opening principles"

    # Début de partie
    if board.fullmove_number <= 4:
        return "chess opening first moves"

    # Milieu de partie approximatif
    if board.fullmove_number <= 12:
        return "chess middlegame plan"

    return "chess positional plan"

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

    fen = state["fen"]
    query = _build_query_from_fen(fen)

    try:
        context = search_chunks(query=query, top_k=5)
    except Exception as e:
        return _add_warning(state, f"Milvus indisponible: {e}")

    return {"context": context or []}


def search_youtube_node(state: AgentState) -> AgentState:
    if state.get("error"):
        return {}

    fen = state["fen"]
    query = _build_query_from_fen(fen)

    try:
        youtube = YouTubeService()
        videos = youtube.search_videos(opening=query, max_results=5)
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