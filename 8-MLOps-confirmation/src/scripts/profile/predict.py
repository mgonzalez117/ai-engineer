# src/scripts/profile_predict.py
"""
Script de profiling pour identifier les goulots d'étranglement
lors de l'inférence (prédiction), simulant le comportement de l'API FastAPI.

Usage:
    poetry run python src/scripts/profile_predict.py

Prérequis:
    - Fichiers de test dans ./.data/extract/:
        * test_sample_application.csv
        * test_sample_bureau.csv
    - Variables d'environnement HF_REPOSITORY, HF_MODEL, HF_PIPELINE
"""

import cProfile
import pstats
import io
from pathlib import Path
from datetime import datetime
import os
import joblib
import pandas as pd
import numpy as np
from huggingface_hub import hf_hub_download

# Variables globales (comme dans l'API)
model = None
pipeline = None
_model_loaded = False


def load_model():
    """
    Charge le modèle et le pipeline depuis Hugging Face (lazy loading).
    Simule exactement le comportement de l'API.
    """
    global model, pipeline, _model_loaded

    if _model_loaded:
        print("/////////////////////////////// Modèle déjà chargé !! //////////")
        return  # déjà chargé

    print("#######################################################")
    print("Chargement du modèle et du pipeline depuis Hugging Face (lazy)...")

    hf_repository = os.getenv("HF_REPOSITORY")
    hf_model = os.getenv("HF_MODEL")
    hf_pipeline = os.getenv("HF_PIPELINE")

    if not all([hf_repository, hf_model, hf_pipeline]):
        raise RuntimeError(
            "Variables d'environnement HF manquantes (HF_REPOSITORY, HF_MODEL, HF_PIPELINE)"
        )

    # Téléchargement et chargement du modèle
    model_path = hf_hub_download(
        repo_id=hf_repository,
        filename=hf_model
    )
    model = joblib.load(model_path)
    print(f"Modèle chargé depuis: {model_path}")

    # Téléchargement et chargement de la pipeline
    pipeline_path = hf_hub_download(
        repo_id=hf_repository,
        filename=hf_pipeline
    )
    pipeline = joblib.load(pipeline_path)
    print(f"Pipeline chargée depuis: {pipeline_path}")

    _model_loaded = True


def load_test_data():
    """Charge les données de test pour le profiling."""
    data_dir = Path('./.data/extract')

    app_path = data_dir / 'test_sample_application.csv'
    bureau_path = data_dir / 'test_sample_bureau.csv'

    if not app_path.exists():
        raise FileNotFoundError(
            f"Fichier {app_path} introuvable. "
            f"Placez vos données de test dans {data_dir}/"
        )

    df_application = pd.read_csv(app_path)
    print(f"Données application chargées: {len(df_application)} lignes")

    # Bureau peut être vide
    if bureau_path.exists():
        df_bureau = pd.read_csv(bureau_path)
        print(f"Données bureau chargées: {len(df_bureau)} lignes")
    else:
        df_bureau = pd.DataFrame()
        print("⚠ Pas de données bureau (fichier absent)")

    return df_application, df_bureau


def simulate_api_prediction(df_application, df_bureau):
    """
    Simule une requête API : lazy loading + prédiction.

    Args:
        df_application: DataFrame avec les données application (1 ligne)
        df_bureau: DataFrame avec les données bureau

    Returns:
        dict: Résultat de la prédiction avec temps d'inférence
    """
    start = datetime.now()

    # Lazy loading (comme dans l'API)
    if model is None or pipeline is None:
        load_model()
    else:
        print("############################ modèle déjà chargé !!!! ##########")

    # Transformation via la pipeline (comme dans l'API)
    X_processed = pipeline.transform(df_application, df_bureau)

    # Prédiction
    pred = int(model.predict(X_processed)[0])

    # Calcul de la probabilité
    if hasattr(model, "predict_proba"):
        proba = float(model.predict_proba(X_processed)[0][1])
    elif hasattr(model, "decision_function"):
        decision = model.decision_function(X_processed)[0]
        proba = float(1 / (1 + np.exp(-decision)))
    else:
        proba = float(pred)

    elapsed_ms = (datetime.now() - start).total_seconds() * 1000

    return {
        'prediction': pred,
        'probability': round(proba, 4),
        'inference_time_ms': round(elapsed_ms, 2)
    }


def run_profiling(df_application, df_bureau, n_iterations=100):
    """
    Exécute n prédictions pour profiler les performances.

    Args:
        df_application: DataFrame avec les données application
        df_bureau: DataFrame avec les données bureau
        n_iterations: Nombre d'itérations pour le profiling
    """
    print(f"\n🔄 Exécution de {n_iterations} prédictions pour profiling...")

    results = []

    for i in range(n_iterations):
        result = simulate_api_prediction(df_application, df_bureau)
        result['iteration'] = i + 1
        results.append(result)

    # Statistiques (en excluant la première itération qui inclut le chargement)
    times = [r['inference_time_ms'] for r in results[1:]]  # Skip première itération

    print(f"\n Statistiques sur {len(times)} prédictions (hors chargement initial):")
    print(f"   - Temps moyen: {np.mean(times):.2f} ms")
    print(f"   - Temps médian: {np.median(times):.2f} ms")
    print(f"   - Temps min: {np.min(times):.2f} ms")
    print(f"   - Temps max: {np.max(times):.2f} ms")
    print(f"   - Écart-type: {np.std(times):.2f} ms")

    # Temps de la première prédiction (avec chargement)
    first_time = results[0]['inference_time_ms']
    print(f"\n⏱  Temps première prédiction (avec lazy loading): {first_time:.2f} ms")

    return results


def main():
    """Fonction principale encapsulant tout le profiling."""

    print("=" * 80)
    print("PROFILING DE L'INFÉRENCE - Simulation du comportement de l'API")
    print("=" * 80)

    # 1. Chargement des données de test
    df_application, df_bureau = load_test_data()

    # 2. Exécution des prédictions (à profiler)
    # Note: Le lazy loading se fera à la première itération
    results = run_profiling(
        df_application=df_application,
        df_bureau=df_bureau,
        n_iterations=100
    )

    print("\n Profiling terminé avec succès")


if __name__ == "__main__":
    # Créer le dossier de sortie pour les résultats de profiling
    Path('.data/profiling').mkdir(parents=True, exist_ok=True)

    # Activer le profiler
    profiler = cProfile.Profile()
    profiler.enable()

    # Exécuter le profiling
    main()

    # Désactiver le profiler
    profiler.disable()

    # Afficher les résultats dans la console
    s = io.StringIO()
    ps = pstats.Stats(profiler, stream=s).sort_stats("cumulative")
    ps.print_stats(50)  # Top 50 fonctions les plus coûteuses

    print("\n" + "=" * 80)
    print("PROFILING RESULTS - Top 50 fonctions par temps cumulé")
    print("=" * 80)
    print(s.getvalue())

    # Sauvegarder le fichier .prof
    prof_file = '.data/profiling/predict_inference.prof'
    profiler.dump_stats(prof_file)
    print(f"\n Fichier de profiling sauvegardé : {prof_file}")