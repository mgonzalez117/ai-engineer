from fastapi import FastAPI, Request, HTTPException, status, Security, Depends
from fastapi.security import HTTPBearer
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from fastapi.responses import RedirectResponse
from starlette.status import HTTP_401_UNAUTHORIZED
import os
import secrets
from src.api import ask, rebuild

API_TOKEN = os.getenv("API_TOKEN")
security = HTTPBearer()

class AuthMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        public_paths = ["/", "/ask", "/status", "/docs", "/openapi.json", "/redoc"]

        if request.url.path in public_paths:
            return await call_next(request)

        try:
            auth_header = request.headers.get("Authorization")
            if not auth_header or not auth_header.startswith("Bearer "):
                raise HTTPException(
                    status_code=HTTP_401_UNAUTHORIZED,
                    detail="Missing or invalid Authorization header",
                    headers={"WWW-Authenticate": "Bearer"},
                )
            token = auth_header[len("Bearer "):]
            if not secrets.compare_digest(token, API_TOKEN):
                raise HTTPException(
                    status_code=HTTP_401_UNAUTHORIZED,
                    detail="Invalid token",
                    headers={"WWW-Authenticate": "Bearer"},
                )
            response = await call_next(request)
            return response

        # handles HTTPException to avoid 500 internal server error and correctly render HTTP response & status code
        except HTTPException as exc:
            return JSONResponse(
                status_code=exc.status_code,
                content={"detail": exc.detail},
                headers=exc.headers or {},
            )

app = FastAPI(
    title="API Chatbot pour OpenAgenda",
    description="Recherche d'évènements publics depuis OpenAgenda",
    version="1.0.0"
)
app.include_router(ask.router)
app.include_router(rebuild.router)
app.add_middleware(AuthMiddleware)

# Rediriger la racine vers Swagger UI
@app.get("/", include_in_schema=False)
async def root():
    return RedirectResponse(url="/redoc")

@app.get("/status")
def status():
    return {"message": "Hello, FastAPI is running 🚀"}

@app.get("/test-auth", dependencies=[Depends(security)])
def test_auth():
    return {"message": f"Your token is valid. Authentication succeeded"}

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(
        status_code=422,
        content={"detail": exc.errors()}
    )