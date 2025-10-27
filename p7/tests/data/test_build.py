# tests/service/test_evaluate_rag.py
import os

from p7.src.service.answer import MISTRAL_TOKEN

os.environ["GIT_PYTHON_REFRESH"] = "quiet"

import json
from pathlib import Path
from typing import List, Dict
import pytest

from datasets import Dataset
from ragas.metrics import answer_relevancy
from ragas import evaluate
from sentence_transformers import SentenceTransformer, util
import pandas as pd

from src.service.answer import answer_question
from langchain_mistralai import ChatMistralAI

TEST_DIR = Path(__file__).parent
TESTSET_PATH = TEST_DIR / "testset.json"
OUT_CSV = TEST_DIR / "rag_evaluation_results.csv"
MODEL_SIM = os.getenv("EMB_MODEL")
MISTRAL_TOKEN = os.getenv("MISTRAL_TOKEN")

# Seuils minimaux
MIN_ANSWER_RELEVANCY = 0.6
MIN_SIMILARITY = 0.5

def load_testset(path: Path) -> List[Dict]:
    lines = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                lines.append(json.loads(line))
    return lines

def get_ragas_judge():
    mistral_key = MISTRAL_TOKEN
    if not mistral_key:
        pytest.skip("MISTRAL_TOKEN manquant pour évaluer avec RAGAS")
    # Modèle Mistral light/medium/large selon ton quota
    return ChatMistralAI(
        model="mistral-small-latest",  # ou "mistral-medium-latest" / "mistral-large-latest"
        api_key=mistral_key,
        temperature=0.0,
    )

@pytest.fixture(scope="module")
def evaluation_results():
    test_items = load_testset(TESTSET_PATH)

    questions, references, answers = [], [], []

    for item in test_items:
        q = item["question"]
        ref = item["reference_answer"]
        ans = answer_question(q, top_k=10)
        questions.append(q)
        references.append(ref)
        answers.append(ans)

    data = {
        "question": questions,
        "answer": answers,
        "ground_truth": references,
    }
    ds = Dataset.from_dict(data)

    # Juge RAGAS = Mistral
    judge = get_ragas_judge()

    result = evaluate(
        ds,
        metrics=[answer_relevancy],
        llm=judge,  # important: évite OpenAI, utilise Mistral
    )
    ragas_scores = result["answer_relevancy"]

    # Similarité embeddings
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
        "answer_relevancy": float(sum(ragas_scores) / len(ragas_scores)),
        "similarity_score": float(sum(cos_sims) / len(cos_sims)),
    }

    print("\n" + "="*60)
    print("RAG — Scores:")
    for k, v in avg_metrics.items():
        print(f"  - {k}: {v:.3f}")
    print(f"\nDétails sauvegardés dans {OUT_CSV}")
    print("="*60 + "\n")

    return avg_metrics, df


def test_testset_exists():
    assert TESTSET_PATH.exists(), f"Le fichier {TESTSET_PATH} n'existe pas"

def test_testset_not_empty():
    test_items = load_testset(TESTSET_PATH)
    assert len(test_items) >= 3, "Le jeu de test doit contenir au moins 3 questions"

def test_answer_relevancy(evaluation_results):
    avg_metrics, _ = evaluation_results
    assert avg_metrics["answer_relevancy"] >= MIN_ANSWER_RELEVANCY, \
        f"Answer relevancy trop faible: {avg_metrics['answer_relevancy']:.3f} < {MIN_ANSWER_RELEVANCY}"

def test_similarity_score(evaluation_results):
    avg_metrics, _ = evaluation_results
    assert avg_metrics["similarity_score"] >= MIN_SIMILARITY, \
        f"Similarity score trop faible: {avg_metrics['similarity_score']:.3f} < {MIN_SIMILARITY}"

def test_all_questions_answered(evaluation_results):
    _, df = evaluation_results
    empty_answers = df[df["generated_answer"].str.strip() == ""]
    assert len(empty_answers) == 0, f"{len(empty_answers)} question(s) sans réponse"

def test_csv_output_created():
    assert OUT_CSV.exists(), f"Le fichier {OUT_CSV} n'a pas été créé"