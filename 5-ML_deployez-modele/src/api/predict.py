from fastapi import APIRouter, Depends, HTTPException
from fastapi.security import HTTPBearer
from sqlalchemy.orm import Session
import pandas as pd
import time
from datetime import datetime
from src.database import get_db
from src.models import PredictLogs
from src.api.prediction_request import PredictionRequest
from lib import model_loader
router = APIRouter()
security = HTTPBearer()

@router.post("/predict", dependencies=[Depends(security)])
def predict(input_data: PredictionRequest, db: Session = Depends(get_db)):
    start_time = time.time()
    status = "error"
    error_message = None
    prediction_result = None

    try:
        pipeline = model_loader.pipeline
        expected = model_loader.expected_inputs

        # Conversion en DataFrame
        X = pd.DataFrame([input_data.model_dump()])

        # Réordonner selon l’ordre du training
        X = X[expected]

        # Prédiction via la pipeline
        y_pred = pipeline.predict(X)
        y_proba = pipeline.predict_proba(X)[0]

        # Résultat
        prediction_result = {
            "prediction": int(y_pred[0]),
            "prediction_label": "Oui" if y_pred[0] == 1 else "Non",
            "probability_quit": round(float(y_proba[1]), 3),
        }

        status = "success"

    except Exception as e:
        status = "error"
        error_message = str(e)
        raise HTTPException(status_code=500, detail=f"Erreur de prédiction: {str(e)}")

    finally:
        # Calcul du temps de traitement
        processing_time_ms = (time.time() - start_time) * 1000

        # Enregistrement en base de données
        log_entry = PredictLogs(
            date=datetime.utcnow(),
            input_payload=input_data.model_dump(),
            prediction_result=prediction_result,
            processing_time_ms=round(processing_time_ms, 2),
            status=status,
            error_message=error_message
        )

        db.add(log_entry)
        db.commit()
        db.refresh(log_entry)

    return prediction_result