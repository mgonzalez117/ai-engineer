import os
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from types import SimpleNamespace

from src.database import SessionLocal
from src.models import PredictLogs
from src.api import predict
from tests.fixtures.predict_payload import base_valid_payload

API_TOKEN = os.environ.get("API_TOKEN", "testtoken")


# Doubles pour le pipeline
class DummyPipelineOK:
    def predict(self, X):
        return [1]

    def predict_proba(self, X):
        return [[0.2, 0.8]]

class DummyPipelineFail:
    """Pipeline qui lève une exception"""
    def predict(self, X):
        raise RuntimeError("Erreur interne du modèle")

    def predict_proba(self, X):
        raise RuntimeError("Erreur interne du modèle")


@pytest.fixture
def auth_headers():
    return {"Authorization": f"Bearer {API_TOKEN}"}


@pytest.fixture
def app_with_ok_pipeline(monkeypatch, base_valid_payload):
    """App avec un pipeline qui fonctionne"""
    dummy = SimpleNamespace(
        pipeline=DummyPipelineOK(),
        expected_inputs=list(base_valid_payload.keys()),
    )
    monkeypatch.setattr("src.api.predict.model_loader", dummy)

    app = FastAPI()
    app.include_router(predict.router)
    return app

@pytest.fixture
def app_with_failing_pipeline(monkeypatch, base_valid_payload):
    """App avec un pipeline qui échoue"""
    dummy = SimpleNamespace(
        pipeline=DummyPipelineFail(),
        expected_inputs=list(base_valid_payload.keys()),
    )
    monkeypatch.setattr("src.api.predict.model_loader", dummy)

    app = FastAPI()
    app.include_router(predict.router)
    return app


def test_predict_success(base_valid_payload, auth_headers, app_with_ok_pipeline):
    client = TestClient(app_with_ok_pipeline)
    resp = client.post("/predict", json=base_valid_payload, headers=auth_headers)

    assert resp.status_code == 200, f"Status={resp.status_code}, body={resp.text}"
    data = resp.json()
    assert "prediction" in data
    assert "prediction_label" in data
    assert "probability_quit" in data
    assert isinstance(data["prediction"], int)
    assert data["prediction_label"] in ("Oui", "Non")
    assert isinstance(data["probability_quit"], (float, int))

    # Vérifier le log en DB
    db = SessionLocal()
    try:
        logs = db.query(PredictLogs).order_by(PredictLogs.date.desc()).all()
        assert len(logs) >= 1
        log = logs[0]  # dernier log
        assert log.status == "success"
        assert log.error_message is None
        assert log.prediction_result is not None
    finally:
        db.close()


def test_predict_failure_validation(auth_headers, app_with_ok_pipeline):
    """Test erreur de validation Pydantic (422)"""
    client = TestClient(app_with_ok_pipeline)
    payload = {"age": 35}  # Payload incomplet
    resp = client.post("/predict", json=payload, headers=auth_headers)

    assert resp.status_code == 422, f"Attendu 422, eu {resp.status_code} ({resp.text})"
    data = resp.json()
    assert "detail" in data

def test_predict_failure_internal_error(base_valid_payload, auth_headers, app_with_failing_pipeline):
    """Test erreur interne du modèle (500)"""
    client = TestClient(app_with_failing_pipeline)
    resp = client.post("/predict", json=base_valid_payload, headers=auth_headers)

    assert resp.status_code == 500, f"Attendu 500, eu {resp.status_code} ({resp.text})"


def test_predict_requires_bearer(base_valid_payload, app_with_ok_pipeline):
    """Test sécurité HTTPBearer sans token"""
    client = TestClient(app_with_ok_pipeline)
    resp = client.post("/predict", json=base_valid_payload)

    assert resp.status_code in (401, 403), f"Attendu 401/403, eu {resp.status_code} ({resp.text})"