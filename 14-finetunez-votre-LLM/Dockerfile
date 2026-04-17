FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir poetry

WORKDIR /app

COPY pyproject.toml poetry.lock* ./

RUN poetry config virtualenvs.create false \
    && poetry install --no-interaction --no-ansi --no-root

COPY src/ /app/src/

ENV PYTHONPATH=/app/src
ENV DATA_DIR=.data
ENV VLLM_INFERENCE_ENDPOINT=/v1/completions
ENV VLLM_MODEL=chsa-lora
ENV REQUEST_TIMEOUT_SECONDS=60
ENV SYSTEM_PROMPT_FILE=/app/src/api/system_prompt.txt

EXPOSE 8000

CMD ["poetry", "run", "uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]