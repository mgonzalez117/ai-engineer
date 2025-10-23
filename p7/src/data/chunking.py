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

    # Si pas de description, retourner un chunk minimal
    if not description.strip():
        text = titre or "Événement sans description"
        return [{
            'text': text,
            'event_id': event.get('uid', ''),
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
        # Ajouter le titre au début du premier chunk seulement
        if idx == 0 and titre:
            chunk_text = f"{titre}. {chunk_text}"

        chunks.append({
            'text': chunk_text,
            'event_id': event.get('uid', ''),
            'chunk_index': idx,
            'metadata': event
        })

    return chunks