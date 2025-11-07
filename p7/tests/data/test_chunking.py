"""
Tests unitaires pour le module test_chunking.py
"""
import pytest
from src.data.chunking import create_chunks_from_event, CHUNK_SIZE, CHUNK_OVERLAP

def test_event_with_description():
    """Test avec un événement ayant une description normale"""
    event = {
        'uid': 'event_123',
        'title_fr': 'Concert de Jazz',
        'longdescription_fr': 'Un magnifique concert de jazz dans le centre-ville. Les meilleurs artistes seront présents.',
        'firstdate_begin': '2025-10-28',
        'location_name': 'Salle Pleyel',
        'location_city': 'Paris',
        'location_address': '252 Rue du Faubourg Saint-Honoré'
    }

    chunks = create_chunks_from_event(event)

    assert len(chunks) > 0
    # métadonnées basiques
    assert chunks[0]['event_id'] == 'event_123'
    assert chunks[0]['chunk_index'] == 0
    # contexte injecté dans le premier chunk
    assert 'Titre: Concert de Jazz' in chunks[0]['text']
    assert 'Date: 2025-10-28' in chunks[0]['text']
    assert 'Lieu: Salle Pleyel' in chunks[0]['text']
    assert 'Paris' in chunks[0]['text']
    # champs additionnels stockés
    assert chunks[0]['titre'] == 'Concert de Jazz'
    assert chunks[0]['date'] == '2025-10-28'
    assert 'Salle Pleyel' in chunks[0]['lieu']


def test_event_without_description():
    """Test avec un événement sans description"""
    event = {
        'uid': 'event_456',
        'title_fr': 'Exposition',
        'longdescription_fr': '',
        'firstdate_begin': '2025-11-05',
        'location_name': 'Musée d’Art',
        'location_city': 'Lyon'
    }

    chunks = create_chunks_from_event(event)

    # Un seul chunk minimal, avec le contexte (titre/date/lieu) si disponible
    assert len(chunks) == 1
    assert chunks[0]['chunk_index'] == 0
    assert chunks[0]['event_id'] == 'event_456'
    # Le texte doit contenir au minimum le contexte disponible
    assert 'Titre: Exposition' in chunks[0]['text']
    assert 'Date: 2025-11-05' in chunks[0]['text']
    assert 'Lieu: Musée d’Art' in chunks[0]['text']
    assert 'Lyon' in chunks[0]['text']
    # Champs additionnels
    assert chunks[0]['titre'] == 'Exposition'
    assert chunks[0]['date'] == '2025-11-05'
    assert 'Musée d’Art' in chunks[0]['lieu']


def test_event_with_nan_description():
    """Test avec une description NaN"""
    event = {
        'uid': 'event_789',
        'title_fr': 'Théâtre',
        'longdescription_fr': float('nan'),
        'firstdate_begin': '2025-12-01',
        'location_name': 'Théâtre National',
        'location_city': 'Marseille'
    }

    chunks = create_chunks_from_event(event)

    assert len(chunks) == 1
    # Contexte présent dans le texte
    assert 'Titre: Théâtre' in chunks[0]['text']
    assert 'Date: 2025-12-01' in chunks[0]['text']
    assert 'Lieu: Théâtre National' in chunks[0]['text']
    assert 'Marseille' in chunks[0]['text']
    # Champs additionnels
    assert chunks[0]['titre'] == 'Théâtre'
    assert chunks[0]['date'] == '2025-12-01'
    assert 'Théâtre National' in chunks[0]['lieu']


def test_event_with_long_description():
    """Test avec une description longue générant plusieurs chunks"""
    long_text = "Lorem ipsum dolor sit amet. " * 50  # Texte long
    event = {
        'uid': 'event_999',
        'title_fr': 'Festival',
        'longdescription_fr': long_text,
        'firstdate_begin': '2025-09-10',
        'location_name': 'Parc Expo',
        'location_city': 'Nantes'
    }

    chunks = create_chunks_from_event(event, chunk_size=100, overlap=20)

    assert len(chunks) > 1
    assert all(chunk['event_id'] == 'event_999' for chunk in chunks)
    assert chunks[0]['chunk_index'] == 0
    assert chunks[1]['chunk_index'] == 1
    # Le premier chunk doit contenir le contexte
    assert 'Titre: Festival' in chunks[0]['text']
    assert 'Date: 2025-09-10' in chunks[0]['text']
    assert 'Lieu: Parc Expo' in chunks[0]['text']
    # ✅ Vérifier que le 2e chunk contient du contenu (pas juste le contexte)
    assert 'Lorem ipsum' in chunks[1]['text']
    # ✅ Vérifier que tous les chunks ont les métadonnées
    assert all(chunk['titre'] == 'Festival' for chunk in chunks)
    assert all(chunk['date'] == '2025-09-10' for chunk in chunks)


def test_chunk_metadata_preserved():
    """Test que les métadonnées sont préservées dans 'metadata'"""
    event = {
        'uid': 'event_111',
        'title_fr': 'Cinéma',
        'longdescription_fr': 'Projection spéciale',
        'location_name': 'UGC Les Halles',
        'location_city': 'Paris',
        'firstdate_begin': '2025-11-01',
        'custom_field': 'foo'
    }

    chunks = create_chunks_from_event(event)

    # Le dict original doit être présent dans 'metadata'
    assert chunks[0]['metadata'] == event
    # Les champs dérivés doivent exister
    assert chunks[0]['titre'] == 'Cinéma'
    assert chunks[0]['date'] == '2025-11-01'
    assert 'UGC Les Halles' in chunks[0]['lieu']