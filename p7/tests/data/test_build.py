"""
Tests unitaires pour le module build.py
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

import pytest
import os
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
from src.data.build import fetch_all_events, build_index


@patch('src.data.build.requests.get')
def test_fetch_all_events_success(mock_get):
    """Test récupération réussie des événements"""
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        'records': [
            {'fields': {'uid': '1', 'title_fr': 'Event 1'}},
            {'fields': {'uid': '2', 'title_fr': 'Event 2'}}
        ],
        'nhits': 2
    }
    mock_get.return_value = mock_response

    df = fetch_all_events()

    assert len(df) == 2
    assert 'uid' in df.columns
    assert df.iloc[0]['uid'] == '1'


@patch('src.data.build.requests.get')
def test_fetch_all_events_empty(mock_get):
    """Test avec aucun événement"""
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {'records': [], 'nhits': 0}
    mock_get.return_value = mock_response

    df = fetch_all_events()

    assert len(df) == 0


@patch('src.data.build.requests.get')
def test_fetch_all_events_api_error(mock_get):
    """Test erreur API"""
    mock_response = MagicMock()
    mock_response.status_code = 500
    mock_get.return_value = mock_response

    df = fetch_all_events()

    assert len(df) == 0


@patch('src.data.build.fetch_all_events')
@patch('src.data.build.SentenceTransformer')
@patch('src.data.build.faiss.write_index')
@patch('builtins.open', create=True)
def test_build_index_with_events(mock_open, mock_faiss_write, mock_model, mock_fetch):
    """Test construction de l'index avec des événements"""
    mock_fetch.return_value = pd.DataFrame([
        {'uid': '1', 'title_fr': 'Event 1', 'longdescription_fr': 'Description 1'}
    ])

    mock_model_instance = MagicMock()
    mock_model_instance.encode.return_value = np.array([[0.1, 0.2, 0.3]])
    mock_model.return_value = mock_model_instance

    build_index()

    mock_model_instance.encode.assert_called_once()
    mock_faiss_write.assert_called_once()


@patch('src.data.build.fetch_all_events')
def test_build_index_no_events(mock_fetch):
    """Test construction de l'index sans événements"""
    mock_fetch.return_value = pd.DataFrame()

    build_index()

    mock_fetch.assert_called_once()