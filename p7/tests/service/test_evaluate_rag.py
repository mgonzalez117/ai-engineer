# tests/service/test_evaluate_rag.py
import os
os.environ["GIT_PYTHON_REFRESH"] = "quiet"

import json
import time
from pathlib import Path
from typing import List, Dict
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
MODEL_SIM = os.getenv("EMB_MODEL")
MISTRAL_TOKEN = os.getenv("MISTRAL_TOKEN")

# Seuils minimaux
MIN_ANSWER_RELEVANCY = 0.6
MIN_SIMILARITY = 0.5

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


def evaluate_with_retry(ds: Dataset, max_retries=3, base_sleep=2.0):
    """
    Tente evaluate(...) avec Mistral en essayant plusieurs modèles
    et un backoff exponentiel en cas de 429. Retourne la liste des scores.
    Lève pytest.skip si aucune tentative ne passe.
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
                # Détecte surcharge/capacity exceeded ou rate-limit
                transient = (
                    "429" in msg
                    or "capacity" in msg.lower()
                    or "rate" in msg.lower()
                    or "limit" in msg.lower()
                    or "temporarily" in msg.lower()
                )
                last_err = e
                if transient and attempt < max_retries:
                    sleep_s = base_sleep * (2 ** (attempt - 1))
                    time.sleep(sleep_s)
                    continue
                # si non-transient ou plus de retries, on tente modèle suivant
                break
        # essaie modèle suivant
    # Si rien n'a marché, on skip proprement RAGAS
    pytest.skip(f"RAGAS (Mistral) indisponible: {last_err}")


@pytest.fixture(scope="module")
def evaluation_results():
    """Fixture qui exécute l'évaluation une seule fois pour tous les tests"""
    test_items = load_testset(TESTSET_PATH)

    questions, references, answers = [], [], []

    for item in test_items:
        q = item["question"]
        ref = item["reference_answer"]
        ans = answer_question(q, top_k=10)
        questions.append(q)
        references.append(ref)
        answers.append(ans)

    ds = Dataset.from_dict({
        "question": questions,
        "answer": answers,
        "ground_truth": references,
    })

    # Tente RAGAS avec retry/bascule de modèle; si indispo => skipped
    try:
        ragas_scores = evaluate_with_retry(ds)
    except pytest.skip.Exception:
        # Si RAGAS skip (pas de clé ou saturation persistante), on garde des 0.0
        # et on ne fera pas l'assert sur answer_relevancy (test sera skipped).
        ragas_scores = [0.0] * len(questions)
        ragas_skipped = True
    else:
        ragas_skipped = False

    # Similarité embeddings (toujours calculée pour garantir la création du CSV)
    st_model = SentenceTransformer(MODEL_SIM)
    emb_ans = st_model.encode(answers, convert_to_tensor=True, normalize_embeddings=True)
    emb_ref = st_model.encode(references, convert_to_tensor=True, normalize_embeddings=True)
    cos_sims = util.cos_sim(emb_ans, emb_ref).diagonal().tolist()

    df = pd.DataFrame({
        "question": questions,
        "reference_answer": references,
        "generated_answer": answers,
        "ragas_answer_relevancy": ragas_scores,
        "similarity_score": cos_sims,
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
    if state["ragas_skipped"]:
        pytest.skip("RAGAS non exécuté (clé absente ou capacité Mistral saturée)")
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