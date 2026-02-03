# LunarLander RL – Projet AI Engineer (P11)

Ce dépôt présente un projet de **Reinforcement Learning** appliqué à l’environnement **LunarLander (Gymnasium)**, avec une démarche MLOps : suivi d’expériences, API d’inférence et interface de démonstration.

## Structure du dépôt

Arborescence (principaux éléments) :

```
.
├── .mlflow/
│   ├── mlartifacts/              # Artefacts MLflow (modèles, fichiers loggés, etc.)
│   └── mlruns/                   # Runs MLflow (métriques, paramètres, tags)
├── experiments/
│   └── experiments.json          # Historique des missions exécutées via la GUI (API -> stockage JSON)
├── src/
│   ├── exercise-1/               # Exercices / itérations intermédiaires
│   ├── exercise-2/
│   ├── exercise-3/
│   └── mission/
│       ├── api/                  # API FastAPI (inférence + endpoints d’expérimentations)
│       ├── gui/                  # Dashboard Streamlit (GUI + visualisations)
│       ├── logs_dqn/             # Logs d’entraînement DQN (selon les notebooks)
│       ├── logs_ppo/             # Logs d’entraînement PPO (selon les notebooks)
│       ├── tensorboard_logs/     # Logs TensorBoard montés dans le service tensorboard
│       ├── videos_dqn/           # Enregistrements vidéo (DQN)
│       ├── videos_ppo/           # Enregistrements vidéo (PPO)
│       ├── 0-mission.ipynb       # Notebook principal (mission)
│       ├── 1-mission-ppo.ipynb   # Variante / essais PPO
│       ├── dqn_lunarlander.zip   # Modèle DQN exporté (Stable-Baselines3)
│       └── ppo_lunarlander.zip   # Modèle PPO exporté (Stable-Baselines3)
├── .env                          # Variables d’environnement (non versionnées selon .gitignore)
├── .env.dist                     # Exemple de variables d’environnement
├── docker-compose.yaml           # Orchestration des services Docker
├── Dockerfile                    # Image commune (API / dashboard / notebooks)
├── poetry.lock                   # Lockfile Poetry
├── pyproject.toml                # Dépendances du projet
└── README.md
```

## Services Docker

Le projet est orchestré avec `docker-compose` et s’appuie sur les services suivants.

### 1) `notebooks` — Jupyter Lab
- **Rôle** : environnement de développement et d’entraînement (notebooks).
- **Port** : `8888`.
- **Spécificité** : configuration de `MLFLOW_TRACKING_URI` pour journaliser les expériences dans MLflow.

### 2) `api` — FastAPI (inférence)
- **Rôle** : exposer le modèle RL via une API REST.
- **Port** : `8000`.
- **Fonctions** :
  - `POST /predict` : prédiction d’une action à partir d’une observation (état).
  - routes `/experiments/*` : enregistrement des missions et agrégats (moyenne, écart-type, etc.) utilisés par le dashboard.
- **Hot reload** : activé avec `uvicorn --reload` (développement).

### 3) `dashboard` — Streamlit (GUI & monitoring)
- **Rôle** : interface de démonstration.
  - exécution d’une mission “live” (appel API à chaque step)
  - visualisation des courbes de récompense, moyenne/écart-type, distribution des actions
  - analyse des décisions selon les “circonstances” (ex. angle, vitesse verticale) à l’échelle d’une mission sélectionnée
- **Port** : `8501`.
- **Connexion** : utilise `API_URL` pour appeler le service `api`.

### 4) `mlflow` — MLflow Tracking Server
- **Rôle** : suivi des expérimentations d’entraînement (paramètres, métriques, artefacts).
- **Port** : `5000`.
- **Persistance** : répertoires montés depuis `.mlflow/`.

### 5) `tensorboard` — TensorBoard
- **Rôle** : visualisation des métriques d’entraînement (logs TensorBoard).
- **Port** : `6006`.
- **Logs** : `src/mission/tensorboard_logs/` monté en lecture seule.

## Démarrage

### Variables d’environnement
Le fichier `.env.dist` fournit un exemple. Les variables attendues sont notamment :
- `MODEL_PATH` : chemin du modèle chargé par l’API (ex. `/app/src/mission/dqn_lunarlander.zip`)
- `MODEL_NAME` : nom de l’environnement Gymnasium (ex. `LunarLander-v3`)

### Lancer l’ensemble des services

```bash
docker-compose up -d
```

### Accès aux interfaces
- Jupyter Lab : http://localhost:8888
- API (Swagger) : http://localhost:8000/docs
- Dashboard : http://localhost:8501
- MLflow : http://localhost:5000
- TensorBoard : http://localhost:6006

## Notes
- Les dossiers `.mlflow/` et `experiments/` assurent la persistance des résultats (runs MLflow et historique des missions).
- En développement, les volumes Docker permettent de prendre en compte les changements de code sans reconstruction d’image (sauf modification des dépendances).
