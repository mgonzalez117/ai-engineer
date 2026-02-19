from fastapi import FastAPI, HTTPException

app = FastAPI(
    title="API",
    description="API",
    version="1.0.0",
    root_path="/v1"
)

@app.get("/healthcheck")
async def health():
    return {"status": "healthy"}