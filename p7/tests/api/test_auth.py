import os
import requests

API_TOKEN = os.getenv("API_TOKEN")
API_URL = os.getenv("API_URL")

# Tests de cas nominaux
def test_protected_route_without_token():
    response = requests.get(f"{API_URL}/test-auth")
    assert response.status_code == 401

def test_protected_route_with_token():
    headers = {"Authorization": f"Bearer {API_TOKEN}"} if API_TOKEN else {}
    response = requests.get(f"{API_URL}/test-auth", headers=headers)
    assert response.status_code == 200
    assert "Your token is valid. Authentication succeeded" in response.json().get("message", "")

# Tests de cas limites
def test_protected_route_with_invalid_token():
    headers = {"Authorization": "Bearer invalid_token_xyz"}
    response = requests.get(f"{API_URL}/test-auth", headers=headers)
    assert response.status_code == 401

def test_protected_route_with_malformed_header():
    headers = {"Authorization": "InvalidFormat"}
    response = requests.get(f"{API_URL}/test-auth", headers=headers)
    assert response.status_code == 401

def test_protected_route_with_empty_token():
    headers = {"Authorization": "Bearer "}
    response = requests.get(f"{API_URL}/test-auth", headers=headers)
    assert response.status_code == 401