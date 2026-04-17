import pandas as pd
import requests
from pathlib import Path
from datetime import datetime
import time
import os

# Configuration
API_URL = os.getenv("API_URL")
API_TOKEN = os.getenv("API_TOKEN")
DATA_DIR = Path(".data")


def load_data():
    """Charge les données application_test et bureau"""
    app_test = pd.read_csv(DATA_DIR / "application_test.csv")
    bureau = pd.read_csv(DATA_DIR / "bureau.csv")
    return app_test, bureau


def get_bureau_records(sk_id_curr, bureau_df):
    """Récupère tous les enregistrements bureau pour un SK_ID_CURR donné"""
    bureau_records = bureau_df[bureau_df['SK_ID_CURR'] == sk_id_curr]

    if bureau_records.empty:
        return []

    # Convertir en liste de dictionnaires
    bureau_list = bureau_records.to_dict('records')

    # Remplacer les NaN par None pour le JSON
    for record in bureau_list:
        for key, value in record.items():
            if pd.isna(value):
                record[key] = None

    return bureau_list


def create_api_payload(app_row, bureau_records):
    """Crée le payload JSON pour l'API"""
    # Convertir la ligne application en dict
    app_dict = app_row.to_dict()

    # Remplacer les NaN par None
    for key, value in app_dict.items():
        if pd.isna(value):
            app_dict[key] = None

    payload = {
        "application_data": app_dict,
        "bureau_data": bureau_records
    }

    return payload


def call_api(payload):
    """Appelle l'API et retourne la réponse"""
    try:
        start_time = time.time()
        response = requests.post(
            API_URL+'/predict',
            json=payload,headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {API_TOKEN}"
            },
            timeout=30
        )
        inference_time = time.time() - start_time

        if response.status_code == 200:
            result = response.json()
            result['inference_time_ms'] = round(inference_time * 1000, 2)
            result['status'] = 'success'
            return result
        else:
            return {
                'status': 'error',
                'status_code': response.status_code,
                'error_message': response.text,
                'inference_time_ms': round(inference_time * 1000, 2)
            }
    except Exception as e:
        return {
            'status': 'error',
            'error_message': str(e),
            'inference_time_ms': None
        }


def generate_production_data(start_index=0, num_records=10):
    """
    Génère des données de production en appelant l'API

    Args:
        start_index: Index de départ dans application_test.csv
        num_records: Nombre d'enregistrements à traiter
        api_url: URL de l'API
    """
    print(f"🚀 Génération de {num_records} prédictions à partir de l'index {start_index}")

    # Charger les données
    print("Chargement des données...")
    app_test, bureau = load_data()

    # Sélectionner les lignes
    end_index = start_index + num_records
    selected_rows = app_test.iloc[start_index:end_index]

    print(f"{len(selected_rows)} lignes sélectionnées (index {start_index} à {end_index - 1})")

    # Préparer les résultats
    results = []
    payloads = []

    # Traiter chaque ligne
    for idx, (_, row) in enumerate(selected_rows.iterrows(), 1):
        sk_id_curr = row['SK_ID_CURR']
        print(f"\n[{idx}/{num_records}] Traitement de SK_ID_CURR: {sk_id_curr}")

        # Récupérer les données bureau
        bureau_records = get_bureau_records(sk_id_curr, bureau)
        print(f"  📋 {len(bureau_records)} enregistrements bureau trouvés")

        # Créer le payload
        payload = create_api_payload(row, bureau_records)
        payloads.append(payload)

        # Appeler l'API
        print(f"Appel de l'API...")
        api_response = call_api(payload)

        # Ajouter les métadonnées
        result = {
            'timestamp': datetime.now().isoformat(),
            'sk_id_curr': int(sk_id_curr),
            'start_index': start_index,
            'record_index': start_index + idx - 1,
            'api_response': api_response
        }

        results.append(result)

        if api_response['status'] == 'success':
            print(f"  ✅ Succès - Temps: {api_response.get('inference_time_ms', 'N/A')} ms")
        else:
            print(f"  ❌ Erreur: {api_response.get('error_message', 'Unknown')}")

        # Pause pour ne pas surcharger l'API
        #time.sleep(0.1)
    return results


if __name__ == "__main__":

    # Génère 100 prédictions à partir de l'index 0
    results = generate_production_data(
        start_index=0,
        num_records=4275
    )