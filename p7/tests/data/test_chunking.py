"""
Tests unitaires pour le module test_chunking.py
"""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))

import pytest
from src.data.chunking import create_chunks_from_event, CHUNK_SIZE, CHUNK_OVERLAP

def test_event_with_description():
    """Test avec un événement ayant une description normale"""
    event = {
        'uid': 'event_123',
        'title_fr': 'Concert de Jazz',
        'longdescription_fr': 'Un magnifique concert de jazz dans le centre-ville. Les meilleurs artistes seront présents.'
    }

    chunks = create_chunks_from_event(event)

    assert len(chunks) > 0
    assert chunks[0]['event_id'] == 'event_123'
    assert 'Concert de Jazz' in chunks[0]['text']
    assert chunks[0]['chunk_index'] == 0


def test_event_without_description():
    """Test avec un événement sans description"""
    event = {
        'uid': 'event_456',
        'title_fr': 'Exposition',
        'longdescription_fr': ''
    }

    chunks = create_chunks_from_event(event)

    assert len(chunks) == 1
    assert chunks[0]['text'] == 'Exposition'
    assert chunks[0]['chunk_index'] == 0


def test_event_with_nan_description():
    """Test avec une description NaN"""
    event = {
        'uid': 'event_789',
        'title_fr': 'Théâtre',
        'longdescription_fr': float('nan')
    }

    chunks = create_chunks_from_event(event)

    assert len(chunks) == 1
    assert chunks[0]['text'] == 'Théâtre'


def test_event_with_long_description():
    """Test avec une description longue générant plusieurs chunks"""
    long_text = "Lorem ipsum dolor sit amet. " * 50  # Texte long
    event = {
        'uid': 'event_999',
        'title_fr': 'Festival',
        'longdescription_fr': long_text
    }

    chunks = create_chunks_from_event(event, chunk_size=100, overlap=20)

    assert len(chunks) > 1
    assert all(chunk['event_id'] == 'event_999' for chunk in chunks)
    assert chunks[0]['chunk_index'] == 0
    assert chunks[1]['chunk_index'] == 1


def test_chunk_metadata_preserved():
    """Test que les métadonnées sont préservées"""
    event = {
        'uid': 'event_111',
        'title_fr': 'Cinéma',
        'longdescription_fr': 'Projection spéciale',
        'location': 'Paris',
        'date': '2025-11-01'
    }

    chunks = create_chunks_from_event(event)

    assert chunks[0]['metadata'] == event