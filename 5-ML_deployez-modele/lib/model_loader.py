# lib/model_loader.py
import joblib
from huggingface_hub import hf_hub_download
import os

repository = os.getenv('HF_REPOSITORY')
filename = os.getenv('HF_MODEL')
token = os.getenv('HF_TOKEN')

# Télécharger le modèle depuis Hugging Face Hub
# (mis en cache localement après le 1er téléchargement)
MODEL_PATH = hf_hub_download(
    repo_id=repository,
    filename=filename,
    token=token
)

# Chargement une seule fois
obj = joblib.load(MODEL_PATH)
pipeline = obj["pipeline"]
expected_inputs = obj["features"]