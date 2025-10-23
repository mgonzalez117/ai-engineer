import os
import pandas as pd
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import pickle
import requests

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
    """Construit l'index FAISS depuis l'API"""
    os.makedirs(INDEX_DIR, exist_ok=True)

    # Récupérer tous les événements depuis l'API
    df = fetch_all_events()

    if len(df) == 0:
        print("Aucun événement récupéré")
        return

    print(f"Construction de l'index avec {len(df)} événements...")

    # Construire l'index
    texts = df.astype(str).apply(lambda row: ' '.join(row.values), axis=1).tolist()

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
        'dataframe': df.to_dict('records')
    }

    # Sauvegarder
    faiss.write_index(index, INDEX_PATH)
    with open(METADATA_PATH, 'wb') as f:
        pickle.dump(metadata, f)

    print(f"✓ Index sauvegardé")
    print(f"✓ Total : {index.ntotal} documents indexés")


if __name__ == '__main__':
    build_index()