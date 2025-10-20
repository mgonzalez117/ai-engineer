FROM python:3.13-slim

WORKDIR /app

# Installer dépendances système de build
RUN apt-get update && apt-get install -y \
    gcc \
    build-essential \
    libpq-dev \
    libssl-dev \
    libffi-dev \
    && rm -rf /var/lib/apt/lists/*

# Installer Poetry
RUN pip install --upgrade pip && \
    pip install poetry

# Configurer Poetry
RUN poetry config virtualenvs.create false

# Copier les fichiers de dépendances
COPY pyproject.toml poetry.lock* ./

# Copier le reste du projet
COPY . .

# Installer les dépendances et le projet
RUN poetry install

EXPOSE 80

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "80"]