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
FILTER_YEARS = os.getenv('FILTER_YEARS')
FILTER_DEPARTMENT = os.getenv('FILTER_DEPARTMENT')
EMB_MODEL = os.getenv('EMB_MODEL')

# Chemins de persistence
INDEX_PATH = os.path.join(INDEX_DIR, 'index.faiss')
METADATA_PATH = os.path.join(INDEX_DIR, 'metadata.pkl')
EVENTS_CSV_PATH = os.path.join(INDEX_DIR, 'events.csv')

BATCH_SIZE = 128

def batch_iter(iterable, n):
    for i in range(0, len(iterable), n):
        yield iterable[i:i + n]


def fetch_all_events():
    """Récupère tous les événements depuis l'API OpenAgenda (Opendatasoft)"""
    params = {
        'limit': 100,
        'offset': 0
    }

    if FILTER_DEPARTMENT and FILTER_YEARS:
        where_clause = 'location_department="' + FILTER_DEPARTMENT + '"'

        years = [year.strip() for year in FILTER_YEARS.split(',') if year.strip()]
        year_conditions = []
        for year in years:
            year_conditions.append(
                f"(firstdate_begin >= '{year}-01-01T00:00:00Z' AND firstdate_begin <= '{year}-12-31T23:59:59Z')")

        where_clause += " AND (" + " OR ".join(year_conditions) + ")"
        params['where'] = where_clause
        print(f"WHERE: {where_clause}")

    all_events = []

    print("Récupération des événements depuis OpenAgenda (Opendatasoft)...")
    while True:
        response = requests.get(API, params=params)

        print(response.url)

        if response.status_code != 200:
            print(f"Erreur API: {response.status_code}")
            break

        data = response.json()
        results = data.get('results', [])  # 'results' au lieu de 'records'
        total_count = data.get('total_count', 0)  # 'total_count' au lieu de 'nhits'

        if not results:
            print("Aucun résultat")
            break

        all_events.extend(results)

        print(f"Récupéré {len(all_events)}/{total_count} événements...")

        if len(all_events) >= total_count:
            break

        params['offset'] += params['limit']

    print(f"Total récupéré : {len(all_events)} événements")

    # Sauvegarder dans un CSV
    if all_events:
        df = pd.DataFrame(all_events)

        if 'uid' in df.columns:
            df = df.drop_duplicates(subset='uid', keep='first')

        date_col = 'firstdate_begin' if 'firstdate_begin' in df.columns else df.columns[0]
        if date_col in df.columns:
            df = df.sort_values(by=date_col)

        df.to_csv(EVENTS_CSV_PATH, index=False)
        print(f"{len(df)} événements uniques sauvegardés dans {EVENTS_CSV_PATH}")

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