FROM python:3.13-slim

WORKDIR /app

# Installer dépendances système
RUN apt-get update && apt-get install -y \
    gcc build-essential libpq-dev libssl-dev libffi-dev \
    && rm -rf /var/lib/apt/lists/*

# Installer Poetry
RUN pip install --upgrade pip && pip install poetry
RUN poetry config virtualenvs.create false

# Copier les fichiers de dépendances
COPY pyproject.toml poetry.lock* ./

# Installer les dépendances
RUN poetry install --no-root --no-interaction --no-ansi

# Copier le code de l'API
COPY ./ ./

# Installer le projet
RUN poetry install --only-root --no-interaction --no-ansi

EXPOSE 80

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "80"]