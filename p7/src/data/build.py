import os
import pandas as pd
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import pickle
import requests
from chunking import create_chunks_from_event

# Configuration depuis les variables d'environnement
INDEX_DIR = os.getenv('INDEX_DIR')
EMB_MODEL = os.getenv('EMB_MODEL')
API = os.getenv('OPENDATASOFT_URL')
API_DATASET = os.getenv('OPENDATASOFT_DATASET')
FILTER_DEPARTMENT = os.getenv('FILTER_DEPARTMENT')
FILTER_YEAR = os.getenv('FILTER_YEAR')

# Chemins de persistence
INDEX_PATH = os.path.join(INDEX_DIR, 'index.faiss')
METADATA_PATH = os.path.join(INDEX_DIR, 'metadata.pkl')


def fetch_all_events():
    """Récupère tous les événements depuis l'API OpenAgenda (Opendatasoft)

    Returns:
        pd.DataFrame: DataFrame de tous les événements
    """
    params = {
        'dataset': API_DATASET,
        'rows': 1000,
        'start': 0
    }

    if FILTER_DEPARTMENT:
        params['refine.location_department'] = FILTER_DEPARTMENT
        print(f"Filtre appliqué : département = {FILTER_DEPARTMENT}")

    if FILTER_YEAR:
        params['refine.firstdate_begin'] = FILTER_YEAR
        print(f"Filtre appliqué : année = {FILTER_YEAR}")

    all_events = []

    print("Récupération des événements depuis OpenAgenda (Opendatasoft)...")
    while True:
        response = requests.get(API, params=params)

        if response.status_code != 200:
            print(f"Erreur API: {response.status_code}")
            break

        data = response.json()
        records = data.get('records', [])

        if not records:
            break

        events = [record['fields'] for record in records]
        all_events.extend(events)

        print(f"Récupéré {len(all_events)} événements...")

        if len(all_events) >= data.get('nhits', 0):
            break

        params['start'] += params['rows']

    print(f"Total récupéré : {len(all_events)} événements")

    return pd.DataFrame(all_events)


def build_index():
    """Construit l'index FAISS depuis l'API avec chunking"""
    os.makedirs(INDEX_DIR, exist_ok=True)

    # Récupérer tous les événements depuis l'API
    df = fetch_all_events()

    if len(df) == 0:
        print("Aucun événement récupéré")
        return

    print(f"Construction de l'index avec {len(df)} événements...")

    # Créer les chunks pour chaque événement
    all_chunks = []
    for idx, event in df.iterrows():
        event_dict = event.to_dict()
        chunks = create_chunks_from_event(event_dict)
        all_chunks.extend(chunks)

    print(f"Total de chunks créés : {len(all_chunks)}")

    # Extraire les textes pour l'embedding
    texts = [chunk['text'] for chunk in all_chunks]

    print(f"Chargement du modèle {EMB_MODEL}...")
    model = SentenceTransformer(EMB_MODEL)

    print("Génération des embeddings...")
    embeddings = model.encode(texts, show_progress_bar=True)
    embeddings = np.array(embeddings).astype('float32')

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)

    metadata = {
        'texts': texts,
        'chunks': all_chunks,
        'num_events': len(df),
        'num_chunks': len(all_chunks)
    }

    # Sauvegarder
    faiss.write_index(index, INDEX_PATH)
    with open(METADATA_PATH, 'wb') as f:
        pickle.dump(metadata, f)

    print(f"✓ Index sauvegardé")
    print(f"✓ Total : {index.ntotal} chunks indexés depuis {len(df)} événements")
    print(f"✓ Moyenne : {len(all_chunks) / len(df):.2f} chunks par événement")


if __name__ == '__main__':
    build_index()