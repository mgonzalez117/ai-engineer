from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from stable_baselines3 import DQN
import numpy as np
import gymnasium as gym
from typing import List
import os

# Import du router des expérimentations
from .experiments import router as experiments_router

app = FastAPI(
    title="LunarLander DQN API",
    description="API pour l'inférence du modèle DQN sur LunarLander-v3",
    version="1.0.0"
)

# Inclusion du router expérimentations
app.include_router(experiments_router, prefix="/experiments", tags=["Expérimentations"])

# Variable globale pour stocker le modèle chargé
model = None
env = None

# Chemin du modèle, nom de l'environnement gymnasium
MODEL_PATH = os.getenv("MODEL_PATH")
MODEL_NAME = os.getenv("MODEL_NAME")


class ObservationInput(BaseModel):
    """
    Observation de l'environnement LunarLander-v3
    8 valeurs : [x, y, vx, vy, angle, angular_velocity, left_leg_contact, right_leg_contact]
    """
    observation: List[float]

    class Config:
        json_schema_extra = {
            "example": {
                "observation": [0.0, 1.5, -0.5, -0.3, 0.1, 0.2, 0.0, 0.0]
            }
        }


class PredictionOutput(BaseModel):
    """Sortie de la prédiction"""
    action: int
    action_name: str


@app.on_event("startup")
async def load_model():
    """Charge le modèle au démarrage de l'API"""
    global model, env

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Le modèle {MODEL_PATH} n'existe pas")

    try:
        model = DQN.load(MODEL_PATH)
        env = gym.make(MODEL_NAME)
        print(f"✅ Modèle chargé depuis {MODEL_PATH}")
    except Exception as e:
        raise RuntimeError(f"Erreur lors du chargement du modèle : {e}")


@app.get("/")
async def root():
    """Endpoint racine"""
    return {
        "message": "API LunarLander DQN",
        "status": "running",
        "model_loaded": model is not None
    }


@app.get("/health")
async def health():
    """Endpoint de santé"""
    if model is None:
        raise HTTPException(status_code=503, detail="Modèle non chargé")
    return {"status": "healthy", "model": MODEL_PATH}


@app.post("/predict", response_model=PredictionOutput)
async def predict(input_data: ObservationInput):
    """
    Prédit l'action à prendre pour une observation donnée

    Actions possibles :
    - 0 : Ne rien faire
    - 1 : Moteur gauche
    - 2 : Moteur principal
    - 3 : Moteur droit
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Modèle non chargé")

    # Vérifier que l'observation a bien 8 valeurs
    if len(input_data.observation) != 8:
        raise HTTPException(
            status_code=400,
            detail=f"L'observation doit contenir 8 valeurs, reçu {len(input_data.observation)}"
        )

    try:
        # Convertir en numpy array
        obs = np.array(input_data.observation, dtype=np.float32)

        # Prédire l'action
        action, _states = model.predict(obs, deterministic=True)
        action = int(action)

        # Mapper l'action à son nom
        action_names = {
            0: "Ne rien faire",
            1: "Moteur gauche",
            2: "Moteur principal",
            3: "Moteur droit"
        }

        return PredictionOutput(
            action=action,
            action_name=action_names.get(action, "Action inconnue")
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur lors de la prédiction : {str(e)}")


@app.get("/model/info")
async def model_info():
    """Retourne les informations sur le modèle chargé"""
    if model is None:
        raise HTTPException(status_code=503, detail="Modèle non chargé")

    return {
        "model_path": MODEL_PATH,
        "algorithm": "DQN",
        "environment": MODEL_NAME,
        "observation_space": "Box(8,)",
        "action_space": "Discrete(4)"
    }