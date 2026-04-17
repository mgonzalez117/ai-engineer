import os
import pytest
from fastapi.testclient import TestClient
from src.api.main import app

API_TOKEN = os.environ.get("API_TOKEN")

@pytest.fixture
def client():
    return TestClient(app)

def test_root_redirects_to_redoc_no_follow(client):
    resp = client.get("/", follow_redirects=False)
    assert resp.status_code in (307, 302)
    assert resp.headers.get("location") == "/redoc"

def test_status(client):
    response = client.get("/status")
    assert response.status_code == 200
    assert response.json() == {"message": "Hello, FastAPI is running 🚀"}

def test_validation_error_handler(client):
    # Envoie une requête invalide sur /predict (ou n'importe quel endpoint avec validation Pydantic)
    resp = client.post("/predict", json={"age": "invalid"}, headers={"Authorization": f"Bearer {API_TOKEN}"})

    assert resp.status_code == 422
    assert "detail" in resp.json()