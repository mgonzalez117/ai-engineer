import os
import faiss
import pickle
import numpy as np
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.embeddings import HuggingFaceEmbeddings

# Configuration
INDEX_DIR = os.getenv('INDEX_DIR')
INDEX_PATH = os.path.join(INDEX_DIR, 'index.faiss')
METADATA_PATH = os.path.join(INDEX_DIR, 'metadata.pkl')
EMB_MODEL = os.getenv('EMB_MODEL', 'sentence-transformers/all-MiniLM-L6-v2')
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

    # Rechercher les chunks les plus pertinents
    distances, indices = index.search(question_embedding, top_k)

    # Récupérer les textes des chunks pertinents
    relevant_chunks = [metadata['texts'][i] for i in indices[0]]
    context = "\n\n".join(relevant_chunks)

    # Créer le prompt
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("user", "Contexte:\n{context}\n\nQuestion: {question}")
    ])

    # Initialiser Mistral
    llm = ChatMistralAI(
        model="mistral-large-latest",
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