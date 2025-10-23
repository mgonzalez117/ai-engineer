import os
import pandas as pd
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import pickle

# Configuration depuis les variables d'environnement
INDEX_DIR = os.getenv('INDEX_DIR')
EMB_MODEL = os.getenv('EMB_MODEL')
CSV_PATH = os.getenv('OPENAGENDA_PATH')

# Chemins de persistance
INDEX_PATH = os.path.join(INDEX_DIR, 'index.faiss')
METADATA_PATH = os.path.join(INDEX_DIR, 'metadata.pkl')


def build_index(force_rebuild=False):
    """Construit ou charge l'index FAISS depuis le CSV

    Args:
        force_rebuild (bool): Si True, reconstruit l'index même s'il existe déjà
    """

    # Créer le répertoire si nécessaire
    os.makedirs(INDEX_DIR, exist_ok=True)

    # Vérifier si l'index existe déjà
    if os.path.exists(INDEX_PATH) and os.path.exists(METADATA_PATH):
        if not force_rebuild:
            print(f"Index existant trouvé dans {INDEX_DIR}")
            print("Utiliser force_rebuild=True pour reconstruire l'index")
            return
        else:
            print(f"Reconstruction forcée de l'index (écrasement des fichiers existants)...")

    print("Construction d'un nouvel index...")

    # Charger le CSV
    df = pd.read_csv(CSV_PATH, sep=";")
    print(f"Chargé {len(df)} lignes depuis {CSV_PATH}")

    # Créer le texte à indexer (adapter selon vos colonnes)
    # Exemple : concaténer toutes les colonnes textuelles
    texts = df.astype(str).apply(lambda row: ' '.join(row.values), axis=1).tolist()

    # Charger le modèle d'embeddings
    print(f"Chargement du modèle {EMB_MODEL}...")
    model = SentenceTransformer(EMB_MODEL)

    # Générer les embeddings
    print("Génération des embeddings...")
    embeddings = model.encode(texts, show_progress_bar=True)
    embeddings = np.array(embeddings).astype('float32')

    # Créer l'index FAISS
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)

    # Sauvegarder l'index
    faiss.write_index(index, INDEX_PATH)
    print(f"Index sauvegardé dans {INDEX_PATH}")

    # Sauvegarder les métadonnées (textes originaux + dataframe)
    metadata = {
        'texts': texts,
        'dataframe': df.to_dict('records')
    }
    with open(METADATA_PATH, 'wb') as f:
        pickle.dump(metadata, f)
    print(f"Métadonnées sauvegardées dans {METADATA_PATH}")

    print(f"✓ Index construit avec {len(texts)} documents")


if __name__ == '__main__':
    # Par défaut, ne reconstruit pas si l'index existe
    # Pour forcer la reconstruction : build_index(force_rebuild=True)
    build_index()