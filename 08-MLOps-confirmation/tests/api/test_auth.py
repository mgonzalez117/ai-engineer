import os
import requests

API_TOKEN = os.getenv("API_TOKEN")
API_URL = os.getenv("API_URL")

# ============================================
# Tests des routes publiques (sans auth)
# ============================================

def test_public_route_status():
    """Route publique /status accessible sans token"""
    response = requests.get(f"{API_URL}/status")
    assert response.status_code == 200
    assert "Hello, FastAPI is running" in response.json().get("message", "")

def test_public_route_root():
    """Route publique / redirige vers /redoc"""
    response = requests.get(f"{API_URL}/", allow_redirects=False)
    assert response.status_code in [307, 302]  # Redirection
    assert "/redoc" in response.headers.get("location", "")

# ============================================
# Tests des routes protégées (avec auth)
# ============================================

def test_protected_route_without_token():
    """Route protégée sans token → 401"""
    response = requests.get(f"{API_URL}/test-auth")
    assert response.status_code == 401
    assert "Missing or invalid Authorization header" in response.json().get("detail", "")

def test_protected_route_with_valid_token():
    """Route protégée avec token valide → 200"""
    headers = {"Authorization": f"Bearer {API_TOKEN}"}
    response = requests.get(f"{API_URL}/test-auth", headers=headers)
    assert response.status_code == 200
    assert "Your token is valid" in response.json().get("message", "")

def test_protected_route_with_invalid_token():
    """Route protégée avec mauvais token → 401"""
    headers = {"Authorization": "Bearer invalid_token_xyz"}
    response = requests.get(f"{API_URL}/test-auth", headers=headers)
    assert response.status_code == 401
    assert "Invalid token" in response.json().get("detail", "")

def test_protected_route_with_malformed_header():
    """Route protégée avec header mal formé → 401"""
    headers = {"Authorization": "InvalidFormat"}
    response = requests.get(f"{API_URL}/test-auth", headers=headers)
    assert response.status_code == 401
    assert "Missing or invalid Authorization header" in response.json().get("detail", "")

def test_protected_route_with_empty_token():
    """Route protégée avec token vide → 401"""
    headers = {"Authorization": "Bearer "}
    response = requests.get(f"{API_URL}/test-auth", headers=headers)
    assert response.status_code == 401
    assert "Invalid token" in response.json().get("detail", "")

# ============================================
# Test du handler de validation (422)
# ============================================

def test_validation_error_handler():
    """Déclenche une erreur de validation pour tester le handler 422"""
    # Suppose que /predict attend un body structuré
    # Envoie un body invalide pour forcer une 422
    headers = {"Authorization": f"Bearer {API_TOKEN}"}
    response = requests.post(
        f"{API_URL}/predict",
        headers=headers,
        json={"champ_invalide": "oops"}  # Adapte selon ton endpoint
    )
    assert response.status_code == 422
    body = response.json()
    assert "detail" in body
    assert isinstance(body["detail"], list)