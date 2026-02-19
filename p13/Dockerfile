FROM python:3.12-slim

RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir poetry

# Utilise /app partout pour être raccord avec le docker-compose
WORKDIR /app

# On copie d'abord les fichiers de dépendances
#COPY pyproject.toml poetry.lock* ./
COPY pyproject.toml ./

# On installe SANS le lock (pour utiliser le tien) et SANS le root
RUN poetry config virtualenvs.create false \
    && poetry install --no-interaction --no-ansi --no-root

# On copie le reste du code
COPY . .

EXPOSE 8888