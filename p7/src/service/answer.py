import os
import faiss
import pickle
import numpy as np
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from datetime import datetime, timezone

# Configuration
INDEX_DIR = os.getenv('INDEX_DIR')
INDEX_PATH = os.path.join(INDEX_DIR, 'index.faiss')
METADATA_PATH = os.path.join(INDEX_DIR, 'metadata.pkl')
EMB_MODEL = os.getenv('EMB_MODEL')
MISTRAL_TOKEN = os.getenv('MISTRAL_TOKEN')

PROMPT_FILE = os.getenv("PROMPT_FILE")
print(PROMPT_FILE)
if not PROMPT_FILE:
    raise ValueError("La fichier de prompt PROMPT_FILE n'est pas défini.")

with open(PROMPT_FILE, "r", encoding="utf-8") as f:
    system_prompt = f.read().strip()


def answer_question(question: str, top_k: int = 10) -> str:
    """
    Répond à une question en utilisant RAG avec Mistral

    Args:
        question: La question posée par l'utilisateur
        top_k: Nombre de chunks à récupérer (défaut: 10)

    Returns:
        La réponse générée par Mistral
    """
    # Charger l'index FAISS et les métadonnées
    index = faiss.read_index(INDEX_PATH)
    with open(METADATA_PATH, 'rb') as f:
        metadata = pickle.load(f)

    # Générer l'embedding de la question
    embeddings_model = HuggingFaceEmbeddings(model_name=EMB_MODEL)
    question_embedding = embeddings_model.embed_query(question)

    # Convertir en numpy array 2D pour FAISS
    question_embedding = np.array([question_embedding], dtype='float32')

    # Rechercher plus de chunks pour pouvoir filtrer
    search_k = max(top_k * 5, 50)
    distances, indices = index.search(question_embedding, search_k)

    # Date/heure actuelle (UTC)
    now_utc = datetime.now(timezone.utc)
    current_date_iso = now_utc.strftime("%Y-%m-%d")

    # Construire candidats avec métadonnées
    candidates = []
    metas = metadata.get('metas', None)
    for rank, i in enumerate(indices[0]):
        meta = metas[i] if metas and i < len(metas) else {}
        text = metadata['texts'][i]
        candidates.append({
            "text": text,
            "meta": meta
        })

    # Filtrer par date uniquement
    filtered = []
    for c in candidates:
        date_iso = c["meta"].get("event_date_iso")
        if date_iso is None or date_iso >= current_date_iso:
            filtered.append(c)

    # Si rien après filtrage, garder les candidats initiaux
    selected = (filtered or candidates)[:top_k]

    # Construire le contexte
    relevant_chunks = [c["text"] for c in selected]
    context = "\n\n".join(relevant_chunks)

    # Créer le prompt
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("user",
         "Contexte (extraits):\n{context}\n\n"
         "Question: {question}\n\n"
         "Rappels obligatoires:\n"
         f"- Date actuelle (UTC): {current_date_iso}\n"
         "- Ne recommande pas d'événements antérieurs à la date actuelle.\n"
         "- Réponds uniquement à partir du contexte fourni.")
    ])

    # Initialiser Mistral
    llm = ChatMistralAI(
        model="mistral-medium-latest",
        api_key=MISTRAL_TOKEN,
        temperature=0.3
    )

    # Générer la réponse
    chain = prompt_template | llm
    response = chain.invoke({
        "context": context,
        "question": question
    })

    return response.content