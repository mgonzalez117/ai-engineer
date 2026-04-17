FROM python:3.13-slim

WORKDIR /app

# Installer dépendances système + cron
RUN apt-get update && apt-get install -y \
    gcc build-essential libpq-dev libssl-dev libffi-dev cron \
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

# Créer le fichier cron : exécution le 1er de chaque mois à 02:00
RUN echo "0 2 1 * * cd /app && /usr/local/bin/poetry run python -m src.drift.monitoring >> /var/log/drift_cron.log 2>&1" > /etc/cron.d/drift_cron

# Donner les bons droits au fichier cron
RUN chmod 0644 /etc/cron.d/drift_cron && crontab /etc/cron.d/drift_cron

# Créer le fichier de log
RUN touch /var/log/drift_cron.log

# Script de démarrage qui lance à la fois l'API et cron
RUN echo '#!/bin/bash\n\
cron\n\
exec uvicorn src.api.main:app --host 0.0.0.0 --port 7860' > /start.sh && \
chmod +x /start.sh

EXPOSE 7860

CMD ["/start.sh"]