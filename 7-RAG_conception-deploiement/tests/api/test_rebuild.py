import pytest
import os
from fastapi.testclient import TestClient
from unittest.mock import patch
from src.api.main import app

client = TestClient(app)

@pytest.fixture
def auth_header():
    token = os.getenv("API_TOKEN")
    return {"Authorization": f"Bearer {token}"}


@patch('src.api.rebuild.build_index')
def test_rebuild_success(mock_build_index, auth_header):
    """Test rebuild avec succès"""
    mock_build_index.return_value = {
        'success': True,
        'message': 'Index construit avec succès',
        'num_events': 150,
        'num_chunks': 450,
        'avg_chunks_per_event': 3.0,
        'embedding_model': 'local:all-MiniLM-L6-v2',
        'dimension': 384
    }

    response = client.put("/rebuild", headers=auth_header)

    assert response.status_code == 200
    assert response.json()['status'] == 'success'
    assert response.json()['details']['num_events'] == 150


@patch('src.api.rebuild.build_index')
def test_rebuild_failure(mock_build_index, auth_header):
    """Test rebuild avec échec"""
    mock_build_index.return_value = {
        'success': False,
        'message': 'Erreur API',
        'num_events': 0,
        'num_chunks': 0
    }

    response = client.put("/rebuild", headers=auth_header)

    assert response.status_code == 500
    assert response.json()['detail']['status'] == 'error'


def test_rebuild_no_auth():
    """Test sans authentification"""
    response = client.put("/rebuild")
    assert response.status_code == 401