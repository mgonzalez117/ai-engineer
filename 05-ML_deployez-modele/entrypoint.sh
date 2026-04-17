#!/bin/sh
set -e

echo "[INIT] Lancement du script d'initialisation de la base de données"
python -m src.init.data_init

echo "[API] Lancement du serveur FastAPI..."
exec uvicorn src.api.main:app --host 0.0.0.0 --port ${API_LISTEN_PORT:-7860}