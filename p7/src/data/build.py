import os
import pandas as pd
import faiss
import numpy as np
import pickle
import requests
from datetime import datetime, timedelta
from src.data.chunking import create_chunks_from_event
from langchain_community.embeddings import HuggingFaceEmbeddings

# Configuration depuis les variables d'environnement
INDEX_DIR = os.getenv('INDEX_DIR')
API = os.getenv('OPENDATASOFT_URL')
API_DATASET = os.getenv('OPENDATASOFT_DATASET')
FILTER_YEARS = os.getenv('FILTER_YEARS')
FILTER_DEPARTMENT = os.getenv('FILTER_DEPARTMENT')
EMB_MODEL = os.getenv('EMB_MODEL')

# Chemins de persistence
INDEX_PATH = os.path.join(INDEX_DIR, 'index.faiss')
METADATA_PATH = os.path.join(INDEX_DIR, 'metadata.pkl')

BATCH_SIZE = 128


def batch_iter(iterable, n):
    for i in range(0, len(iterable), n):
        yield iterable[i:i + n]


def fetch_all_events():
    """Récupère tous les événements depuis l'API OpenAgenda (Opendatasoft)"""
    params = {
        'dataset': API_DATASET,
        'rows': 1000,
        'start': 0
    }

    if FILTER_DEPARTMENT:
        params['refine.location_department'] = FILTER_DEPARTMENT
        print(f"Filtre appliqué : département = {FILTER_DEPARTMENT}")

    # Filtre par années
    if FILTER_YEARS:
        years = [year.strip() for year in FILTER_YEARS.split(',')]

        # Construire les plages de dates pour chaque année
        date_ranges = []
        for year in years:
            start_date = f"{year}/01/01"
            end_date = f"{year}/12/31"
            date_ranges.append(f"firstdate_begin:[{start_date} TO {end_date}]")

        # Combiner avec OR
        params['q'] = ' OR '.join(date_ranges)
        print(f"Filtre appliqué : années = {', '.join(years)}")

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
    try:
        os.makedirs(INDEX_DIR, exist_ok=True)

        # Récupérer tous les événements depuis l'API
        df = fetch_all_events()

        if len(df) == 0:
            print("Aucun événement récupéré")
            return {
                'success': False,
                'message': 'Aucun événement récupéré',
                'num_events': 0,
                'num_chunks': 0
            }

        # Créer les chunks pour chaque événement
        all_chunks = []
        for _, event in df.iterrows():
            event_dict = event.to_dict()
            chunks = create_chunks_from_event(event_dict)
            all_chunks.extend(chunks)

        # Extraire les textes pour l'embedding
        texts = [chunk['text'] for chunk in all_chunks]

        # Génération des embeddings avec HuggingFaceEmbeddings
        local_model_name = EMB_MODEL
        used_model = f"local:{local_model_name}"

        embeddings_model = HuggingFaceEmbeddings(
            model_name=local_model_name,
            show_progress=True
        )

        all_embeddings = embeddings_model.embed_documents(texts)

        embeddings = np.array(all_embeddings, dtype='float32')

        dimension = embeddings.shape[1]

        # Index IVF pour recherche rapide sur grands volumes
        nlist = min(int(np.sqrt(len(embeddings))), len(embeddings) // 39)
        quantizer = faiss.IndexFlatL2(dimension)
        index = faiss.IndexIVFFlat(quantizer, dimension, nlist)
        index.train(embeddings)
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

        return {
            'success': True,
            'message': 'Index construit avec succès',
            'num_events': len(df),
            'num_chunks': len(all_chunks),
            'avg_chunks_per_event': round(len(all_chunks) / len(df), 2),
            'embedding_model': used_model,
            'dimension': dimension
        }

    except Exception as e:
        return {
            'success': False,
            'message': f'Erreur : {str(e)}',
            'num_events': 0,
            'num_chunks': 0
        }