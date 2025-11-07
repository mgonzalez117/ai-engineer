"""
Tests unitaires pour le module build.py
"""

import pytest
import os
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock, mock_open
from src.data.build import fetch_all_events, build_index


@patch('src.data.build.requests.get')
@patch('pandas.DataFrame.to_csv')  # ✅ Mock to_csv pour éviter la sauvegarde
def test_fetch_all_events_success(mock_to_csv, mock_get):
    """Test récupération réussie des événements"""
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        'results': [
            {'uid': '1', 'title_fr': 'Event 1'},
            {'uid': '2', 'title_fr': 'Event 2'}
        ],
        'total_count': 2
    }
    mock_response.url = 'http://mock-url.com'
    mock_get.return_value = mock_response

    df = fetch_all_events()

    assert len(df) == 2
    assert 'uid' in df.columns
    assert df.iloc[0]['uid'] == '1'


@patch('src.data.build.requests.get')
@patch('pandas.DataFrame.to_csv')  # ✅ Mock to_csv
def test_fetch_all_events_empty(mock_to_csv, mock_get):
    """Test avec aucun événement"""
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        'results': [],
        'total_count': 0
    }
    mock_response.url = 'http://mock-url.com'
    mock_get.return_value = mock_response

    df = fetch_all_events()

    assert len(df) == 0


@patch('src.data.build.requests.get')
def test_fetch_all_events_api_error(mock_get):
    """Test erreur API"""
    mock_response = MagicMock()
    mock_response.status_code = 500
    mock_response.url = 'http://mock-url.com'
    mock_get.return_value = mock_response

    df = fetch_all_events()

    assert len(df) == 0


@patch('src.data.build.fetch_all_events')
@patch('src.data.build.HuggingFaceEmbeddings')
@patch('src.data.build.faiss.write_index')
@patch('builtins.open', new_callable=mock_open)
def test_build_index_with_events(mock_file, mock_faiss_write, mock_embeddings, mock_fetch):
    """Test construction de l'index avec des événements"""
    mock_fetch.return_value = pd.DataFrame([
        {'uid': '1', 'title_fr': 'Event 1', 'longdescription_fr': 'Description 1'}
    ])

    # Mock de l'instance HuggingFaceEmbeddings
    mock_embeddings_instance = MagicMock()
    mock_embeddings_instance.embed_documents.return_value = [[0.1, 0.2, 0.3]]
    mock_embeddings.return_value = mock_embeddings_instance

    result = build_index()

    # Vérifications
    mock_embeddings_instance.embed_documents.assert_called_once()
    mock_faiss_write.assert_called_once()
    assert result['success'] == True
    assert result['num_events'] == 1
    assert result['num_chunks'] > 0


@patch('src.data.build.fetch_all_events')
def test_build_index_no_events(mock_fetch):
    """Test construction de l'index sans événements"""
    mock_fetch.return_value = pd.DataFrame()

    result = build_index()

    mock_fetch.assert_called_once()
    assert result['success'] == False
    assert result['num_events'] == 0
    assert result['num_chunks'] == 0