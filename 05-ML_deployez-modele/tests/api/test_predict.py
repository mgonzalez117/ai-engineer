import os
import pytest
import requests
import json
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from src.database import get_db_engine, SessionLocal
from src.models import PredictLogs
from tests.fixtures.predict_payload import base_valid_payload

API_URL = os.environ.get("API_URL")
API_TOKEN = os.environ.get("API_TOKEN")
DATABASE_URL = os.environ.get("DATABASE_URL")

engine = get_db_engine()

@pytest.fixture
def auth_headers():
    return {"Authorization": f"Bearer {API_TOKEN}"}

def _url(path: str) -> str:
    return API_URL.rstrip("/") + path

def _assert_business_result(result, expected=None, prob_tol=0.01):
    # structure et types
    assert "prediction" in result, "Le résultat doit contenir prediction"
    assert "prediction_label" in result, "Le résultat doit contenir prediction_label"
    assert "probability_quit" in result, "Le résultat doit contenir probability_quit"
    assert isinstance(result["prediction"], int), "Le résultat de la prédiction doit être un entier (0 ou 1)"
    assert result["prediction_label"] in ("Oui", "Non"), "Le résultat label de prédiction doit être Oui ou Non"
    assert isinstance(result["probability_quit"], (float, int)), "Le résultat de la probabilité de quitter doit être un float ou un entier"

    # valeurs attendues
    if expected:
        if "prediction" in expected:
            assert result["prediction"] == expected["prediction"]
        if "prediction_label" in expected:
            assert result["prediction_label"] == expected["prediction_label"]
        if "probability_quit" in expected:
            assert abs(float(result["probability_quit"]) - float(expected["probability_quit"])) < prob_tol

def test_predict_success(base_valid_payload, auth_headers):
    resp = requests.post(_url("/predict"), json=base_valid_payload, headers=auth_headers, timeout=10)
    assert resp.status_code == 200, f"Status={resp.status_code}, body={resp.text}"
    data = resp.json()

    # Vérification minimaliste des valeurs métiers pour s'assurer qu'on ne dénature pas le modèle lors de son utilisation
    expected = {
       "prediction": 0,
       "prediction_label": "Non",
       "probability_quit": 0.027,
    }

    _assert_business_result(data, expected)

    # Vérifier le log en DB
    db = SessionLocal()
    try:
        logs = db.query(PredictLogs).all()
        assert len(logs) >= 1
        log = logs[-1]  # dernier log
        assert log.status == "success", "log db : status doit être en Succès"
        assert log.error_message is None, "log db : error_message ne doit pas être None"

        # Vérifier le payload d'entrée loggué
        assert log.input_payload is not None, "log db : input_payload ne doit pas être None"
        assert log.input_payload == base_valid_payload, f"input_payload DB différent du payload envoyé.\nDB: {log.payload}\nSent: {base_valid_payload}"

        # Vérification du résultat loggué
        assert log.prediction_result is not None, "log db : prediction_result ne doit pas être None"

        if isinstance(log.prediction_result, str):
            db_result = json.loads(log.prediction_result)
        else:
            db_result = log.prediction_result

        _assert_business_result(db_result, expected)

    finally:
        db.close()

def test_predict_failure(auth_headers):
    # Payload incomplet pour provoquer une erreur
    payload = {"age": 35}
    resp = requests.post(_url("/predict"), json=payload, headers=auth_headers, timeout=10)
    assert resp.status_code == 422, f"Attendu 422, eu {resp.status_code} ({resp.text})"
    data = resp.json()
    assert "detail" in data

def test_predict_requires_bearer(base_valid_payload):
    resp = requests.post(_url("/predict"), json=base_valid_payload, timeout=10)
    assert resp.status_code in (401, 403), f"Attendu 401/403, eu {resp.status_code} ({resp.text})"