from pydantic import BaseModel, Field

class ObservationInput(BaseModel):
    """
    Observation de l'environnement LunarLander-v2
    8 valeurs : [x, y, vx, vy, angle, angular_velocity, left_leg_contact, right_leg_contact]
    """
    observation: list[float]

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


class ExperimentIn(BaseModel):
    """Données d'entrée pour enregistrer une expérimentation"""
    run_name: str = Field(default="streamlit_demo")
    nb_steps: int
    total_reward: float
    terminated: bool
    truncated: bool
    duration_s: float


class ExperimentOut(ExperimentIn):
    """Expérimentation enregistrée avec métadonnées"""
    experiment_id: str
    created_at: float  # epoch seconds