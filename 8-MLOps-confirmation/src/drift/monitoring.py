import sys
from pathlib import Path

import pandas as pd
from sqlalchemy import text
from evidently import Report
from evidently.presets import DataDriftPreset

from src.data.models import DriftRun, DriftFeatureMetric
from src.data.database import get_db
from datetime import datetime

# Ajuste ce chemin à ton projet si besoin
ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT_DIR))

from src.data.database import get_db

# Config
DATA_DIR = ROOT_DIR / ".data"
TRAIN_PATH = DATA_DIR / "application_train.csv"
WINDOW_DAYS = 365
REPORT_OUTPUT = DATA_DIR / "drift" / "report.html"

def extract_prod_data() -> pd.DataFrame:
    """Extrait les données prod depuis la table predict_logs."""
    db = next(get_db())

    query = f"""
        SELECT input_payload->'application_data' AS data
        FROM public.predict_logs
        WHERE status = 'success'
          AND date >= NOW() - INTERVAL '{WINDOW_DAYS} days'
    """

    rows = db.execute(text(query)).fetchall()
    db.close()

    if not rows:
        return pd.DataFrame()

    data = [row[0] for row in rows]
    return pd.DataFrame(data)


def load_reference_data() -> pd.DataFrame:
    """Charge les données d'entraînement comme référence."""
    print(f"Chargement des données de référence depuis {TRAIN_PATH}")
    return pd.read_csv(TRAIN_PATH)


def save_drift_to_db(result_dict: dict):
    from src.data.database import get_db
    from datetime import datetime

    db = next(get_db())

    try:
        metrics = result_dict.get("metrics", [])

        # Créer le run
        drift_run = DriftRun(
            date=datetime.utcnow(),
            dataset_drift=any(float(m["value"]) > m["config"].get("threshold", 0.1)
                              for m in metrics
                              if m["config"].get("type") == "evidently:metric_v2:ValueDrift"),
            drift_score=next((float(m["value"]["share"]) for m in metrics
                              if m["config"].get("type") == "evidently:metric_v2:DriftedColumnsCount"), None)
        )
        db.add(drift_run)
        db.flush()

        # Ajouter les features
        for m in metrics:
            if m["config"].get("type") != "evidently:metric_v2:ValueDrift":
                continue

            val = float(m["value"])
            threshold = m["config"].get("threshold", 0.1)

            db.add(DriftFeatureMetric(
                run_id=drift_run.id,
                feature_name=m["config"]["column"],
                drift_detected=val > threshold,
                drift_score=val,
                stattest_name=m["config"].get("method"),
            ))

        db.commit()

    except Exception as e:
        db.rollback()
        raise
    finally:
        db.close()

def generate_drift_report(reference_data: pd.DataFrame, current_data: pd.DataFrame) -> None:
    """Génère un rapport HTML de drift avec Evidently."""

    # Créer le dossier reports s'il n'existe pas
    REPORT_OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    # Aligner les colonnes entre référence et production
    common_cols = list(set(reference_data.columns) & set(current_data.columns))

    if not common_cols:
        print("Aucune colonne commune entre les données de référence et de production!")
        return

    print(f"Colonnes communes détectées: {len(common_cols)}")

    # Colonnes à exclure de l'analyse de drift
    EXCLUDE_COLS = ['SK_ID_CURR']

    # Exclure les colonnes
    common_cols = [col for col in common_cols if col not in EXCLUDE_COLS]
    print(
        f"Colonnes exclues: {[col for col in EXCLUDE_COLS if col in set(reference_data.columns) & set(current_data.columns)]}")

    reference_subset = reference_data[common_cols]
    current_subset = current_data[common_cols]

    print("Génération du rapport de drift...")

    # Ici : API Evidently "nouvelle" avec Report + metric_preset
    report = Report(
        metrics=[
            DataDriftPreset(),
        ]
    )

    eval = report.run(
        reference_data=reference_subset,
        current_data=current_subset,
    )

    # Cette version‑là de Report a bien save_html
    eval.save_html(str(REPORT_OUTPUT))

    result_dict = eval.dict()
    save_drift_to_db(result_dict)

    print(f"Rapport de drift généré: {REPORT_OUTPUT}")


def main():
    """Point d'entrée principal."""
    print("=" * 60)
    print("ANALYSE DE DRIFT DU MODÈLE")
    print("=" * 60)

    # 1. Charger les données de référence (train)
    reference_data = load_reference_data()
    print(f"Données de référence: {reference_data.shape}")

    # 2. Extraire les données de production
    print(f"\nExtraction des données de production ({WINDOW_DAYS} derniers jours)...")
    current_data = extract_prod_data()

    if current_data.empty:
        print("Aucune donnée de production trouvée!")
        return

    print(f"Données de production: {current_data.shape} (lignes: {len(current_data)})")

    # 3. Générer le rapport de drift
    generate_drift_report(reference_data, current_data)

    print("\n" + "=" * 60)
    print("ANALYSE TERMINÉE")
    print("=" * 60)


if __name__ == "__main__":
    main()