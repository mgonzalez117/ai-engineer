from fastapi import APIRouter, HTTPException, status, Depends
from pydantic import BaseModel
from typing import List
from datetime import datetime
import logging
from sqlalchemy.orm import Session
import os
import joblib
from huggingface_hub import hf_hub_download
import pandas as pd
import numpy as np
from imblearn.pipeline import Pipeline as ImbPipeline
from xgboost import XGBClassifier

from src.api.validation import ApplicationTest, Bureau
from src.data.models import PredictLogs
from src.data.database import get_db

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/predict",
    tags=["prediction"],
)

class PredictionRequest(BaseModel):
    application_data: ApplicationTest
    bureau_data: List[Bureau]

class PredictionResponse(BaseModel):
    SK_ID_CURR: int
    prediction: int
    probability: float
    inference_time_ms: float


# Variables globales
model = None
pipeline = None
_model_loaded = False   # petit flag pour éviter de recharger à chaque requête


def load_model():
    """
    Charge le modèle et le pipeline depuis Hugging Face.
    Peut être appelée au startup OU en lazy loading.
    """
    global model, pipeline, _model_loaded

    if _model_loaded:
        logger.info("/////////////////////////////// Modele déjà chargé !! //////////")
        print("/////////////////////////////// Modele déjà chargé !! //////////")
        return  # déjà chargé

    try:
        print("#######################################################")
        print("Chargement du modèle et du pipeline depuis Hugging Face (lazy)...")

        hf_repository = os.getenv("HF_REPOSITORY")
        hf_model = os.getenv("HF_MODEL")
        hf_pipeline = os.getenv("HF_PIPELINE")

        if not all([hf_repository, hf_model, hf_pipeline]):
            raise RuntimeError(
                "Variables d'environnement HF manquantes (HF_REPOSITORY, HF_MODEL, HF_PIPELINE)"
            )

        logger.info(f"Téléchargement du modèle {hf_model} depuis {hf_repository}...")
        model_path = hf_hub_download(
            repo_id=hf_repository,
            filename=hf_model
        )
        logger.info(f"Modèle téléchargé à: {model_path}")
        model = joblib.load(model_path)
        logger.info("Modèle chargé avec succès.")

        logger.info(f"Téléchargement de la pipeline {hf_pipeline} depuis {hf_repository}...")
        pipeline_path = hf_hub_download(
            repo_id=hf_repository,
            filename=hf_pipeline
        )
        logger.info(f"Pipeline téléchargée à: {pipeline_path}")
        pipeline = joblib.load(pipeline_path)
        logger.info("Pipeline chargée avec succès.")

        _model_loaded = True

    except Exception as e:
        logger.exception(f"Erreur lors du chargement du modèle/pipeline: {e}")
        # On remonte une 500 pour que le client comprenne que c'est côté serveur
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Erreur lors du chargement du modèle/pipeline: {str(e)}",
        )


@router.post("/", response_model=PredictionResponse, status_code=status.HTTP_200_OK)
async def predict(request: PredictionRequest, db: Session = Depends(get_db)):
    """
    Endpoint de prédiction.
    Lazy loading : si le modèle n'est pas chargé, on le charge à la première requête.
    """
    start = datetime.now()

    try:
        # Lazy loading ici
        if model is None or pipeline is None:
            load_model()
        else:
            print("############################ modèle déjà chargé !!!! ##########")

        input_payload = request.model_dump()

        # ======= Préparation des données =======
        application_dict = request.application_data.model_dump()
        bureau_list = [b.model_dump() for b in request.bureau_data]

        df_application = pd.DataFrame([application_dict])
        df_bureau = pd.DataFrame(bureau_list) if bureau_list else pd.DataFrame()

        # pipeline est un ApplicationBureauPipeline
        X_processed = pipeline.transform(df_application, df_bureau)

        pred = int(model.predict(X_processed)[0])

        if hasattr(model, "predict_proba"):
            proba = float(model.predict_proba(X_processed)[0][1])
        elif hasattr(model, "decision_function"):
            decision = model.decision_function(X_processed)[0]
            proba = float(1 / (1 + np.exp(-decision)))
        else:
            proba = float(pred)

        elapsed_ms = (datetime.now() - start).total_seconds() * 1000

        response_obj = PredictionResponse(
            SK_ID_CURR=request.application_data.SK_ID_CURR,
            prediction=pred,
            probability=round(proba, 4),
            inference_time_ms=round(elapsed_ms, 2),
        )

        # Log succès
        log_entry = PredictLogs(
            input_payload=input_payload,
            prediction_result=response_obj.model_dump(),
            processing_time_ms=round(elapsed_ms, 2),
            status="success",
            error_message=None,
        )
        db.add(log_entry)
        db.commit()

        return response_obj

    except HTTPException:
        # Log erreur HTTP explicite
        elapsed_ms = (datetime.now() - start).total_seconds() * 1000
        log_entry = PredictLogs(
            input_payload=request.model_dump(),
            prediction_result=None,
            processing_time_ms=round(elapsed_ms, 2),
            status="error",
            error_message="HTTPException",
        )
        db.add(log_entry)
        db.commit()
        raise

    except Exception as e:
        logger.exception("Erreur lors de la prédiction")

        elapsed_ms = (datetime.now() - start).total_seconds() * 1000
        log_entry = PredictLogs(
            input_payload=request.model_dump(),
            prediction_result=None,
            processing_time_ms=round(elapsed_ms, 2),
            status="error",
            error_message=str(e),
        )
        db.add(log_entry)
        db.commit()

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Erreur lors de la prédiction: {str(e)}",
        )