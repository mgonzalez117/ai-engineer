# tests/service/test_evaluate_rag.py
import os
os.environ["GIT_PYTHON_REFRESH"] = "quiet"

import json
import time
from pathlib import Path
from typing import List, Dict
import random
import pytest

from datasets import Dataset
from ragas.metrics import answer_relevancy
from ragas import evaluate
from sentence_transformers import SentenceTransformer, util
import pandas as pd

from langchain_mistralai import ChatMistralAI
from src.service.answer import answer_question

TEST_DIR = Path(__file__).parent
TESTSET_PATH = TEST_DIR / "testset.json"
OUT_CSV = TEST_DIR / "rag_evaluation_results.csv"
MODEL_SIM = os.getenv("EMB_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
MISTRAL_TOKEN = os.getenv("MISTRAL_TOKEN") or os.getenv("MISTRAL_API_KEY")

# Seuils minimaux (configurables)
MIN_ANSWER_RELEVANCY = float(os.getenv("RAGAS_MIN_ANSWER_RELEVANCY", "0.6"))
MIN_SIMILARITY = float(os.getenv("RAGAS_MIN_SIMILARITY", "0.5"))

# Limiter la taille du jeu de test pour réduire la charge
MAX_ITEMS = int(os.getenv("RAGAS_MAX_ITEMS", "3"))

# Retries/backoff configurables
RAGAS_MAX_RETRIES = int(os.getenv("RAGAS_MAX_RETRIES", "5"))
RAGAS_BASE_SLEEP = float(os.getenv("RAGAS_BASE_SLEEP", "2.5"))

MISTRAL_MODELS_TRY = [
    "mistral-small-latest",
    "mistral-medium-latest",
    "mistral-large-latest",
]


def load_testset(path: Path) -> List[Dict]:
    lines = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                lines.append(json.loads(line))
    return lines


def build_mistral(model_name: str):
    return ChatMistralAI(
        model=model_name,
        api_key=MISTRAL_TOKEN,
        temperature=0.0,
    )


def evaluate_with_retry(ds: Dataset, max_retries=RAGAS_MAX_RETRIES, base_sleep=RAGAS_BASE_SLEEP):
    """
    Tente evaluate(...) avec Mistral en essayant plusieurs modèles
    et un backoff exponentiel + jitter en cas de 429. Retourne la liste des scores.
    Lève pytest.skip si aucune tentative ne passe (sera géré par fallback local).
    """
    if not MISTRAL_TOKEN:
        pytest.skip("MISTRAL_TOKEN/MISTRAL_API_KEY manquant pour évaluer avec RAGAS")

    last_err = None
    for model_name in MISTRAL_MODELS_TRY:
        for attempt in range(1, max_retries + 1):
            try:
                judge = build_mistral(model_name)
                result = evaluate(ds, metrics=[answer_relevancy], llm=judge)
                return result["answer_relevancy"]
            except Exception as e:
                msg = str(e)
                transient = (
                    "429" in msg
                    or "capacity" in msg.lower()
                    or "rate" in msg.lower()
                    or "limit" in msg.lower()
                    or "temporarily" in msg.lower()
                    or "overload" in msg.lower()
                )
                last_err = e
                if transient and attempt < max_retries:
                    sleep_s = base_sleep * (2 ** (attempt - 1))
                    sleep_s = sleep_s * (0.8 + 0.4 * random.random())
                    time.sleep(sleep_s)
                    continue
                # non-transient ou plus de retries => on tente le modèle suivant
                break
        # essaie modèle suivant
    # Si rien n'a marché, on skip proprement RAGAS (le fallback local prendra le relai)
    pytest.skip(f"RAGAS (Mistral) indisponible: {last_err}")


# NEW: Fallback local très simple pour approximer answer_relevancy via similarité cosinus
def local_relevancy_fallback(answers, references, emb_model_name: str):
    """
    Approxime la 'answer_relevancy' par la similarité cosinus embeddings.
    Valeurs entre 0 et 1, compatible avec l'attente RAGAS pour un seuil simple.
    """
    st_model = SentenceTransformer(emb_model_name)
    emb_ans = st_model.encode(answers, convert_to_tensor=True, normalize_embeddings=True)
    emb_ref = st_model.encode(references, convert_to_tensor=True, normalize_embeddings=True)
    sims = util.cos_sim(emb_ans, emb_ref).diagonal().tolist()
    # clamp au cas où
    sims = [max(0.0, min(1.0, float(s))) for s in sims]
    return sims


@pytest.fixture(scope="module")
def evaluation_results():
    """Fixture qui exécute l'évaluation une seule fois pour tous les tests"""
    test_items_full = load_testset(TESTSET_PATH)
    test_items = test_items_full[:MAX_ITEMS]

    questions, references, answers = [], [], []

    for item in test_items:
        q = item["question"]
        ref = item["reference_answer"]
        ans = answer_question(q, top_k=5)  # un peu moins coûteux
        questions.append(q)
        references.append(ref)
        answers.append(ans)

    ds = Dataset.from_dict({
        "question": questions,
        "answer": answers,
        "ground_truth": references,
    })

    # Tente RAGAS; si indispo, bascule en fallback local
    ragas_skipped = False
    try:
        ragas_scores = evaluate_with_retry(ds)
    except pytest.skip.Exception:
        ragas_skipped = True
        ragas_scores = local_relevancy_fallback(answers, references, MODEL_SIM)

    # Similarité embeddings (toujours calculée)
    st_model = SentenceTransformer(MODEL_SIM)
    emb_ans = st_model.encode(answers, convert_to_tensor=True, normalize_embeddings=True)
    emb_ref = st_model.encode(references, convert_to_tensor=True, normalize_embeddings=True)
    cos_sims = util.cos_sim(emb_ans, emb_ref).diagonal().tolist()
    cos_sims = [float(s) for s in cos_sims]

    df = pd.DataFrame({
        "question": questions,
        "reference_answer": references,
        "generated_answer": answers,
        "ragas_answer_relevancy": ragas_scores,
        "similarity_score": cos_sims,
        "ragas_mode": ["mistral" if not ragas_skipped else "local_fallback"] * len(questions),
    })
    df.to_csv(OUT_CSV, index=False, encoding="utf-8")

    avg_metrics = {
        "answer_relevancy": float(sum(ragas_scores) / len(ragas_scores)) if ragas_scores else 0.0,
        "similarity_score": float(sum(cos_sims) / len(cos_sims)) if cos_sims else 0.0,
    }

    return {"avg": avg_metrics, "df": df, "ragas_skipped": ragas_skipped}


def test_testset_exists():
    assert TESTSET_PATH.exists(), f"Le fichier {TESTSET_PATH} n'existe pas"


def test_testset_not_empty():
    test_items = load_testset(TESTSET_PATH)
    assert len(test_items) >= 3, "Le jeu de test doit contenir au moins 3 questions"


def test_answer_relevancy(evaluation_results):
    state = evaluation_results
    # NEW: ne plus skip même si Mistral indispo; on a un fallback local
    assert state["avg"]["answer_relevancy"] >= MIN_ANSWER_RELEVANCY, \
        f"Answer relevancy trop faible: {state['avg']['answer_relevancy']:.3f} < {MIN_ANSWER_RELEVANCY}"


def test_similarity_score(evaluation_results):
    state = evaluation_results
    assert state["avg"]["similarity_score"] >= MIN_SIMILARITY, \
        f"Similarity score trop faible: {state['avg']['similarity_score']:.3f} < {MIN_SIMILARITY}"


def test_all_questions_answered(evaluation_results):
    state = evaluation_results
    df = state["df"]
    empty_answers = df[df["generated_answer"].str.strip() == ""]
    assert len(empty_answers) == 0, f"{len(empty_answers)} question(s) sans réponse"


def test_csv_output_created():
    assert OUT_CSV.exists(), f"Le fichier {OUT_CSV} n'a pas été créé"