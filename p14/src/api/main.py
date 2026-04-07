from __future__ import annotations

import hashlib
import json
import os
import re
import time
import unicodedata
import uuid
from datetime import UTC, datetime
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
TRACE_LOG_PATH = Path(os.getenv("TRACE_LOG_PATH", "artifacts/api/traces/interactions.jsonl"))

SYSTEM_PROMPT_FILE = Path(
    os.getenv("SYSTEM_PROMPT_FILE", str(Path(__file__).with_name("system_prompt.txt")))
)

ALLOWED_URGENCY = {"immediat", "tres urgent", "urgent", "differable"}
URGENCY_ALIASES = {
    "immediate": "immediat",
    "immediat": "immediat",
    "urgence maximale": "immediat",
    "urgence vitale": "immediat",
    "critique": "immediat",
    "vital": "immediat",
    "tres urgent": "tres urgent",
    "tres urgente": "tres urgent",
    "very urgent": "tres urgent",
    "urgent": "urgent",
    "urgence moderee": "urgent",
    "moderee": "urgent",
    "modere": "urgent",
    "differable": "differable",
    "differe": "differable",
    "differee": "differable",
    "non urgent": "differable",
}
UNSAFE_PATTERNS = [
    "http://",
    "https://",
    "www.",
    "prescription",
    "dosage",
    "medicament",
]


TRIAGE_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "niveau_urgence": {
            "type": "string",
            "enum": ["immediat", "tres urgent", "urgent", "differable"],
        },
        "orientation": {"type": "string"},
        "justification": {"type": "string"},
        "garde_fou_active": {"type": "boolean"},
    },
    "required": ["niveau_urgence", "orientation", "justification", "garde_fou_active"],
    "additionalProperties": False,
}


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


def now_utc_iso() -> str:
    return datetime.now(UTC).isoformat()


def write_trace(record: dict[str, Any]) -> None:
    try:
        TRACE_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with TRACE_LOG_PATH.open("a", encoding="utf-8") as file:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception:
        # Logging must never block inference responses.
        pass


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


def normalize_urgency(value: str) -> str:
    cleaned = unicodedata.normalize("NFKD", value or "")
    cleaned = cleaned.encode("ascii", "ignore").decode("ascii")
    cleaned = re.sub(r"[^a-zA-Z ]+", " ", cleaned).lower()
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return URGENCY_ALIASES.get(cleaned, cleaned)


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
        compact = re.sub(r",\s*}", "}", json_text)
        compact = re.sub(r",\s*]", "]", compact)
        try:
            data = json.loads(compact)
        except json.JSONDecodeError:
            return fallback_triage("json invalide")

    if not isinstance(data, dict):
        return fallback_triage("format non objet")

    urgency = normalize_urgency(str(data.get("niveau_urgence", "")))
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
    started_at = time.perf_counter()
    status_code = 200
    error_message = ""
    triage: TriageOutput | None = None

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
        "guided_json": TRIAGE_JSON_SCHEMA,
    }

    try:
        async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT_SECONDS) as client:
            response = await client.post(url, headers=headers, json=body)
            if response.status_code == 400 and body.get("guided_json") is not None:
                body_without_guidance = dict(body)
                body_without_guidance.pop("guided_json", None)
                response = await client.post(url, headers=headers, json=body_without_guidance)
            response.raise_for_status()
            data = response.json()

        triage = parse_json_triage(extract_output(data))
        return GenerateResponse(request_id=request_id, model=VLLM_MODEL, triage=triage)
    except httpx.TimeoutException as exc:
        status_code = 504
        error_message = "vLLM timeout"
        raise HTTPException(status_code=504, detail=error_message) from exc
    except httpx.HTTPStatusError as exc:
        status_code = 502
        error_message = f"vLLM returned HTTP {exc.response.status_code}"
        raise HTTPException(status_code=502, detail=error_message) from exc
    except httpx.HTTPError as exc:
        status_code = 502
        error_message = "vLLM unavailable"
        raise HTTPException(status_code=502, detail=error_message) from exc
    finally:
        latency_ms = round((time.perf_counter() - started_at) * 1000, 2)
        trace_record = {
            "timestamp": now_utc_iso(),
            "request_id": request_id,
            "endpoint": "/v1/generate",
            "status_code": status_code,
            "latency_ms": latency_ms,
            "model": VLLM_MODEL,
            "prompt_chars": len(payload.prompt),
            "prompt_sha256": hashlib.sha256(payload.prompt.encode("utf-8")).hexdigest(),
            "max_tokens": payload.max_tokens,
            "temperature": payload.temperature,
            "niveau_urgence": triage.niveau_urgence if triage else None,
            "garde_fou_active": triage.garde_fou_active if triage else None,
            "error": error_message,
        }
        write_trace(trace_record)
