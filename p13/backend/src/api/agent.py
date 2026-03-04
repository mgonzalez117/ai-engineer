from __future__ import annotations

from fastapi import APIRouter, HTTPException

from agent import get_agent_graph

router = APIRouter(prefix="/api/v1", tags=["agent"])

graph = get_agent_graph()


@router.get("/agent/{fen:path}")
async def agent(fen: str):
    result = await graph.ainvoke({"fen": fen})
    final = result.get("final")

    if not final:
        raise HTTPException(status_code=500, detail="Agent: réponse vide")

    if final.get("error") and str(final["error"]).startswith("FEN invalide"):
        raise HTTPException(status_code=400, detail=final["error"])

    return final