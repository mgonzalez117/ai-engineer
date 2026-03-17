from fastapi import FastAPI

app = FastAPI(
    title="API",
    description="API",
    version="1.0.0",
)

@app.get("/healthcheck")
async def health():
    return {"status": "healthy"}