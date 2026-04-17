import os
import pandas as pd
from enum import Enum

class Assets(Enum):
    SIRH_UNIFIED = './assets/sirh_unified.csv'
    SIRH_NORMALIZED = './assets/sirh_normalized.csv'

class Steps(Enum):
    _1_ANALYSIS = 0
    _1b_DIFFERENTIAL = 1
    _2_PREPARE = 2
    _3_CLASSIFICATION = 3

    def get_asset(self):
        mapping = {
            Steps._1b_DIFFERENTIAL: Assets.SIRH_UNIFIED,
            Steps._2_PREPARE: Assets.SIRH_UNIFIED,
            Steps._3_CLASSIFICATION: Assets.SIRH_NORMALIZED,
        }
        return mapping.get(self)

def load_dataframe(step=Steps._1_ANALYSIS):
    asset = step.get_asset()
    if asset:
        df = load_file(asset.value)
        assert df is not None, "Erreur : Le DataFrame n'a pas été chargé correctement (valeur None), assurez-vous de bien avoir exécuté les étapes précédentes du notebook."
        return df
    else:
        print(f"Aucun asset défini pour l'étape: {step}")
        return None

def load_file(file_path):
    df = None

    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
    else:
        print(f"✗ Le fichier {file_path} n'existe pas")

    return df
