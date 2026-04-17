import pytest
import faiss
import numpy as np
import time
import os

INDEX_PATH = os.path.join(os.getenv("INDEX_DIR"), "index.faiss")

@pytest.fixture
def faiss_index():
    """Charge l'index FAISS pour les tests."""
    return faiss.read_index(INDEX_PATH)


def test_index_not_empty(faiss_index):
    """Vérifie que l'index contient des vecteurs."""
    assert faiss_index.ntotal > 0, "L'index est vide"
    print(f"\n✓ Index contient {faiss_index.ntotal} vecteurs")


def test_index_is_trained(faiss_index):
    """Vérifie que l'index est entraîné."""
    assert faiss_index.is_trained, "L'index n'est pas entraîné"
    print(f"\n✓ Index entraîné (dimension: {faiss_index.d})")


def test_search_speed(faiss_index):
    """Vérifie que les recherches sont rapides (< 50ms par requête)."""
    n_queries = 50
    k = 5

    # Générer des requêtes aléatoires
    query_vectors = np.random.random((n_queries, faiss_index.d)).astype('float32')

    # Mesurer le temps
    start = time.time()
    distances, indices = faiss_index.search(query_vectors, k)
    elapsed = time.time() - start

    avg_time_ms = (elapsed / n_queries) * 1000

    print(f"\n✓ Temps moyen par requête: {avg_time_ms:.2f}ms")
    assert avg_time_ms < 50, f"Recherche trop lente: {avg_time_ms:.2f}ms"


def test_search_returns_results(faiss_index):
    """Vérifie que les recherches retournent des résultats valides."""
    query = np.random.random((1, faiss_index.d)).astype('float32')
    distances, indices = faiss_index.search(query, 5)

    # Vérifier que les indices sont valides
    assert np.all(indices >= 0), "Indices négatifs trouvés"
    assert np.all(indices < faiss_index.ntotal), "Indices hors limites"
    print(f"\n✓ Recherche retourne {len(indices[0])} résultats valides")