from fastapi import FastAPI

from api.moves import router as moves_router
from api.evaluate import router as eval_router
from api.vector_search import router as vector_router
from api.videos import router as videos_router
from api.agent import router as agent_router

app = FastAPI(
    title="API",
    description="API",
    version="1.0.0",
    root_path="/v1"
)

app.include_router(moves_router)
app.include_router(eval_router)
app.include_router(vector_router)
app.include_router(videos_router)
app.include_router(agent_router)

@app.get("/healthcheck")
async def health():
    return {"status": "healthy"}