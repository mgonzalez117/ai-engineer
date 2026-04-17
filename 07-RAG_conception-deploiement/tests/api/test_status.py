import os
import requests

API_URL = os.getenv("API_URL")

def test_status():
    response = requests.get(f"{API_URL}/status")
    assert response.status_code == 200
    assert response.json() == {"message": "Hello, FastAPI is running 🚀"}
