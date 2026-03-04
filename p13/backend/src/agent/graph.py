from __future__ import annotations

from langgraph.graph import StateGraph, END

from agent.state import AgentState
from agent.nodes import (
    validate_fen_node,
    get_theoretical_moves_node,
    evaluate_stockfish_node,
    compose_response_node,
)

# Singleton
_GRAPH = None

def _route_after_moves(state: AgentState) -> str:
    # Si FEN invalide => on compose directement (final avec error)
    if state.get("error"):
        return "compose"

    moves = state.get("theory_moves", [])
    return "compose" if moves else "evaluate"


def _build_graph():
    g = StateGraph(AgentState)

    g.add_node("validate", validate_fen_node)
    g.add_node("moves", get_theoretical_moves_node)
    g.add_node("evaluate", evaluate_stockfish_node)
    g.add_node("compose", compose_response_node)

    g.set_entry_point("validate")
    g.add_edge("validate", "moves")

    g.add_conditional_edges(
        "moves",
        _route_after_moves,
        {
            "evaluate": "evaluate",
            "compose": "compose",
        },
    )

    g.add_edge("evaluate", "compose")
    g.add_edge("compose", END)

    return g.compile()


def get_agent_graph():
    global _GRAPH
    if _GRAPH is None:
        _GRAPH = _build_graph()
    return _GRAPH