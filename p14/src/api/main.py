from __future__ import annotations

import json
import os
import re
import uuid
from pathlib import Path
from typing import Any, Literal

import httpx
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

VLLM_BASE_URL = os.getenv("VLLM_BASE_URL", "http://localhost:8001")
VLLM_INFERENCE_ENDPOINT = os.getenv("VLLM_INFERENCE_ENDPOINT", "/v1/completions")
VLLM_MODEL = os.getenv("VLLM_MODEL", "Qwen/Qwen3-1.7B")
VLLM_API_KEY = os.getenv("VLLM_API_KEY", "")
REQUEST_TIMEOUT_SECONDS = float(os.getenv("REQUEST_TIMEOUT_SECONDS", "45"))

SYSTEM_PROMPT_FILE = Path(
    os.getenv("SYSTEM_PROMPT_FILE", str(Path(__file__).with_name("system_prompt.txt")))
)

ALLOWED_URGENCY = {"immediat", "tres urgent", "urgent", "differable"}
UNSAFE_PATTERNS = [
    "http://",
    "https://",
    "www.",
    "prescription",
    "dosage",
    "medicament",
]


class GenerateRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=8000)
    max_tokens: int = Field(default=180, ge=1, le=2048)
    temperature: float = Field(default=0.3, ge=0.0, le=2.0)


class TriageOutput(BaseModel):
    niveau_urgence: Literal["immediat", "tres urgent", "urgent", "differable"]
    orientation: str
    justification: str
    garde_fou_active: bool = False


class GenerateResponse(BaseModel):
    request_id: str
    model: str
    triage: TriageOutput


def load_system_prompt() -> str:
    if SYSTEM_PROMPT_FILE.exists():
        content = SYSTEM_PROMPT_FILE.read_text(encoding="utf-8").strip()
        if content:
            return content

    return (
        "Tu es un agent de triage des urgences. "
        "Tu reponds uniquement en JSON strict avec les cles: "
        "niveau_urgence, orientation, justification, garde_fou_active."
    )


TRIAGE_SYSTEM_PROMPT = load_system_prompt()

app = FastAPI(
    title="P14 API",
    description="FastAPI endpoint proxying vLLM for inference.",
    version="1.0.0",
)


def build_model_prompt(user_prompt: str) -> str:
    return (
        f"{TRIAGE_SYSTEM_PROMPT}\n\n"
        f"Donnees patient:\n{user_prompt.strip()}\n\n"
        "Reponds UNIQUEMENT en JSON strict sur une seule ligne.\n"
        "Format exact:\n"
        '{"niveau_urgence":"immediat|tres urgent|urgent|differable",'
        '"orientation":"...",'
        '"justification":"...",'
        '"garde_fou_active":false}\n'
        "Ne mets aucun texte avant ou apres le JSON."
    )


def extract_output(vllm_json: dict[str, Any]) -> str:
    choices = vllm_json.get("choices", [])
    if not choices:
        return ""

    first = choices[0] or {}
    text = first.get("text")
    if isinstance(text, str):
        return text.strip()

    message = first.get("message", {}) or {}
    content = message.get("content")
    if isinstance(content, str):
        return content.strip()

    return ""


def fallback_triage(reason: str) -> TriageOutput:
    return TriageOutput(
        niveau_urgence="tres urgent",
        orientation="evaluation medicale immediate au service des urgences",
        justification=f"Reponse de securite: {reason}. Validation clinicien requise.",
        garde_fou_active=True,
    )


def parse_json_triage(raw_output: str) -> TriageOutput:
    text = (raw_output or "").strip()
    if not text:
        return fallback_triage("sortie vide")

    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        return fallback_triage("json absent")

    json_text = match.group(0)
    try:
        data = json.loads(json_text)
    except json.JSONDecodeError:
        return fallback_triage("json invalide")

    if not isinstance(data, dict):
        return fallback_triage("format non objet")

    urgency = str(data.get("niveau_urgence", "")).strip().lower()
    orientation = str(data.get("orientation", "")).strip()
    justification = str(data.get("justification", "")).strip()

    if urgency not in ALLOWED_URGENCY:
        return fallback_triage("niveau_urgence invalide")
    if not orientation or not justification:
        return fallback_triage("champs manquants")

    combined = f"{orientation}\n{justification}".lower()
    if any(pattern in combined for pattern in UNSAFE_PATTERNS):
        return fallback_triage("contenu hors cadre triage")

    return TriageOutput(
        niveau_urgence=urgency,  # type: ignore[arg-type]
        orientation=orientation,
        justification=justification,
        garde_fou_active=bool(data.get("garde_fou_active", False)),
    )


@app.get("/healthcheck")
async def health() -> dict[str, str]:
    return {"status": "healthy", "backend": "vllm"}


@app.post("/v1/generate", response_model=GenerateResponse)
async def generate(payload: GenerateRequest) -> GenerateResponse:
    request_id = str(uuid.uuid4())
    url = f"{VLLM_BASE_URL.rstrip('/')}/{VLLM_INFERENCE_ENDPOINT.lstrip('/')}"

    headers = {"Content-Type": "application/json"}
    if VLLM_API_KEY:
        headers["Authorization"] = f"Bearer {VLLM_API_KEY}"

    body = {
        "model": VLLM_MODEL,
        "prompt": build_model_prompt(payload.prompt),
        "max_tokens": payload.max_tokens,
        "temperature": payload.temperature,
        "top_p": 0.9,
        "repetition_penalty": 1.15,
        "stop": ["\n\nDonnees patient:", "<FIN>"],
    }

    try:
        async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT_SECONDS) as client:
            response = await client.post(url, headers=headers, json=body)
            response.raise_for_status()
            data = response.json()
    except httpx.TimeoutException as exc:
        raise HTTPException(status_code=504, detail="vLLM timeout") from exc
    except httpx.HTTPStatusError as exc:
        raise HTTPException(status_code=502, detail=f"vLLM returned HTTP {exc.response.status_code}") from exc
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=502, detail="vLLM unavailable") from exc

    triage = parse_json_triage(extract_output(data))
    return GenerateResponse(request_id=request_id, model=VLLM_MODEL, triage=triage)
