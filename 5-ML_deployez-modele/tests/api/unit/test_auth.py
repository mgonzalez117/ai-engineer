import os
import pytest
from fastapi.testclient import TestClient
from src.api.main import app

API_TOKEN = os.environ.get("API_TOKEN")

@pytest.fixture
def client():
    return TestClient(app)

# Cas nominaux et limites sur /test-auth qui est protégé par Depends(security)
def test_protected_route_without_token(client):
    resp = client.get("/test-auth")
    assert resp.status_code == 401
    # Le middleware répond "Missing or invalid Authorization header" sur les routes protégées
    assert "Missing" in resp.json().get("detail", "") or "Not authenticated" in resp.json().get("detail", "")


def test_protected_route_with_token(client):
    headers = {"Authorization": f"Bearer {API_TOKEN}"}
    resp = client.get("/test-auth", headers=headers)
    assert resp.status_code == 200
    assert "Your token is valid. Authentication succeeded" in resp.json().get("message", "")


def test_protected_route_with_invalid_token(client):
    headers = {"Authorization": "Bearer invalid_token_xyz"}
    resp = client.get("/test-auth", headers=headers)
    assert resp.status_code == 401
    assert resp.json().get("detail") in ("Invalid token", "Missing or invalid Authorization header")


def test_protected_route_with_malformed_header(client):
    headers = {"Authorization": "InvalidFormat"}
    resp = client.get("/test-auth", headers=headers)
    assert resp.status_code == 401
    assert "Missing or invalid Authorization header" in resp.json().get("detail", "")


def test_protected_route_with_empty_token(client):
    headers = {"Authorization": "Bearer "}
    resp = client.get("/test-auth", headers=headers)
    assert resp.status_code == 401
    # soit header invalide, soit token vide qui ne matche pas compare_digest
    assert resp.json().get("detail") in ("Missing or invalid Authorization header", "Invalid token")