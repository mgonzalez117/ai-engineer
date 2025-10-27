"""
Module de chunking pour les descriptions d'événements
Utilise uniquement LangChain
"""

from typing import List, Dict
from langchain_text_splitters import RecursiveCharacterTextSplitter


# Paramètres de chunking
CHUNK_SIZE = 200
CHUNK_OVERLAP = 50

def create_chunks_from_event(event: Dict, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[Dict]:
    """Crée des chunks à partir d'un événement

    Args:
        event: Dictionnaire représentant un événement
        chunk_size: Taille maximale d'un chunk
        overlap: Chevauchement entre chunks

    Returns:
        List[Dict]: Liste de chunks avec métadonnées
    """
    # Récupérer la description longue
    description = event.get('longdescription_fr', '')

    # Convertir en string si c'est pas déjà le cas (gérer les NaN)
    if not isinstance(description, str):
        description = ''

    # Récupérer le titre
    titre = event.get('title_fr', '')
    if not isinstance(titre, str):
        titre = ''

    # Récupérer la date
    date_start = event.get('firstdate_begin', event.get('date_start', ''))
    if not isinstance(date_start, str):
        date_start = ''

    # Récupérer le lieu
    location_name = event.get('location_name', '')
    location_city = event.get('location_city', '')
    location_address = event.get('location_address', '')

    # Construire le lieu complet
    lieu_parts = [location_name, location_address, location_city]
    lieu = ', '.join([p for p in lieu_parts if p and isinstance(p, str)])

    # Construire le contexte (titre + date + lieu)
    contexte_parts = []
    if titre:
        contexte_parts.append(f"Titre: {titre}")
    if date_start:
        contexte_parts.append(f"Date: {date_start}")
    if lieu:
        contexte_parts.append(f"Lieu: {lieu}")

    contexte = '. '.join(contexte_parts)

    # Si pas de description, retourner un chunk minimal avec contexte
    if not description.strip():
        text = contexte if contexte else "Événement sans description"
        return [{
            'text': text,
            'event_id': event.get('uid', ''),
            'titre': titre,
            'date': date_start,
            'lieu': lieu,
            'chunk_index': 0,
            'metadata': event
        }]

    # Chunker avec LangChain
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", "! ", "? ", "; ", ", ", " ", ""]
    )

    # Chunker la description
    text_chunks = text_splitter.split_text(description)

    # Créer les chunks avec métadonnées
    chunks = []
    for idx, chunk_text in enumerate(text_chunks):
        # Ajouter le contexte (titre + date + lieu) au début du premier chunk
        if idx == 0 and contexte:
            chunk_text = f"{contexte}. {chunk_text}"

        chunks.append({
            'text': chunk_text,
            'event_id': event.get('uid', ''),
            'titre': titre,
            'date': date_start,
            'lieu': lieu,
            'chunk_index': idx,
            'metadata': event
        })

    return chunks