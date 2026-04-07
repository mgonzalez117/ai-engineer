# AI Engineer - Finetunez votre propre LLM
## Installation du projet

### Pré-requis : Docker

Le projet se base sur une architecture containerisée avec docker.
`docker` doit être installé sur la machine hôte.

### Initialiser le projet : Make

`make` doit être installé sur votre hôte.
Lancez la commande suivante pour initialiser le projet : 

```
make install
```

Cela va télécharger les datasets si cela n'est pas déjà fait, dans le dossier suivant : `./data/dataset/raw`.

## API d'inference (FastAPI + vLLM)

### Architecture
- `p14-api` expose les endpoints HTTP (`/healthcheck`, `/v1/generate`).
- `p14-vllm` sert le modele avec vLLM (base model + LoRA).
- Flux: client -> `p14-api` -> `p14-vllm` -> `p14-api` -> client.

### Conteneur `p14-vllm`
- Image: `vllm/vllm-openai:latest`.
- Port: `8001` (host) -> `8000` (container).
- LoRA active via `--enable-lora`.
- LoRA chargee depuis Hugging Face via:
  - `HF_TOKEN`
  - `VLLM_LORA_REPO` (defaut: `MGonzalez117/chsa-finetuning`)
  - `VLLM_LORA_ALIAS` (defaut: `chsa-lora`)

### Conteneur `p14-api`
- Build local depuis `Dockerfile`.
- Port expose: `8000`.
- Prompt systeme lu depuis `src/api/system_prompt.txt`.
- Variables de routage inference:
  - `VLLM_BASE_URL`
  - `VLLM_INFERENCE_ENDPOINT` (defaut: `/v1/inference`)
  - `VLLM_MODEL` (defaut recommande: `chsa-lora`)
  - `VLLM_API_KEY` (optionnel)

### Demarrage local (vLLM + API)
```bash
cp .env.dist .env
# renseigner HF_TOKEN dans .env

docker compose up -d p14-vllm p14-api
```

### Verification rapide
```bash
curl http://localhost:8000/healthcheck

curl -X POST http://localhost:8000/v1/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt":"Douleur thoracique brutale avec dyspnee.","max_tokens":120}'
```

### Deploiement CI de l'image `p14-api`
Le workflow GitHub Actions `test-p14.yml` build et push l'image Docker `p14-api` uniquement si:
- les tests `p14` passent,
- le push est sur `main`,
- des fichiers API/image ont change (`p14/Dockerfile`, `p14/src/api/**`, `p14/pyproject.toml`, `p14/poetry.lock`).

Secrets GitHub requis (Repository -> Settings -> Secrets and variables -> Actions):
- `DOCKERHUB_USERNAME`
- `DOCKERHUB_TOKEN`

Sans ces secrets, l'etape de build/push de l'image est refusee.

### Utiliser un serveur vLLM externe
Configurer dans `.env`:
- `VLLM_BASE_URL` (ex: URL Runpod)
- `VLLM_INFERENCE_ENDPOINT`
- `VLLM_MODEL`
- `VLLM_API_KEY` si necessaire

Puis lancer uniquement:
```bash
docker compose up -d p14-api
```



## Datasets

Nous utilisons les datasets suivants : 

### 1. MediQAl

Lien : https://huggingface.co/datasets/ANR-MALADES/MediQAl

#### Nature du dataset
MediQAl est un dataset français de QA médicale conçu pour évaluer à la fois le rappel factuel (retrouver une information précise) et le raisonnement clinique.
Il contient 32 603 questions sur 41 sujets médicaux et trois configurations : mcqu, mcqm et oeq. Chaque question est annotée Understanding ou Reasoning

/!\  MediQAl n’est pas un dataset homogène. Il contient en réalité :

* des questions ouvertes (oeq)
* des QCM (mcqu, mcqm)

#### Apport métier

Cette partie apporte :
* des QCM médicaux français, parfois avec cas clinique, avec une vraie information métier importante : spécialité médicale, type cognitif, et type de QCM (unique ou multiple).
* de la QA ouverte française, souvent contextualisée par un cas clinique.

#### Champs
* `id` : identifiant de la question
* `clinical_case` : description du cas clinique lorsque présent
* `question` : question
* `answer_a`, `answer_b`, `answer_c`, `answer_d`, `answer_e` : réponses possibles
* `correct_answers` : différentes réponses possibles (notation "A,B,C" par exemple)
* `task` : classe type (QCM / OPEN)
* `medical_subject` : domaine médical du cas / spécialité
* `question_type` : Type de question (compréhension, raisonnement)

### 2. MedQuAD
Lien : https://huggingface.co/datasets/keivalya/MedQuad-MedicalQnADataset

#### Nature du dataset
MedQuAD est un dataset anglais de question-réponse ouverte. La version Hugging Face affichée contient un split train d’environ 16,4k lignes et trois champs visibles : qtype, Question, Answer. qtype a 16 classes.

#### Apport métier
MedQuAD apporte des exemples de QA ouverte en anglais. Il est plus approprié pour du SFT de type open QA, pas du QCM et pas du preference learning.

#### Champs

* `qtype` : Catégorie de la question (16 classes)
* `question` : question posée
* `answer` : réponse textuelle ouverte

### 3. FrenchMedMCQA
Lien :  https://huggingface.co/datasets/nthngdy/frenchmedmcqa

#### Nature du dataset
FrenchMedMCQA est un dataset français de QCM médicaux. Il contient 3 105 questions issues d’examens réels du diplôme de spécialisation médicale en pharmacie, avec des questions à réponse simple et multiple.

#### Apport métier
FrenchMedMCQA apporte des QCM médicaux en français, propres et homogènes, sans cas clinique explicite dans la structure affichée. C’est une très bonne source pour la famille MCQ.

#### Champs
* `id` : identifiant de la question
* `question` : question posée
* `answer_a`, `answer_b`, `answer_c`, `answer_d`, `answer_e` : réponses possibles
* `correct_answers` : différentes réponses possibles (notation index de la réponse, ex: 2 = C)
* `number_correct_answers` : nombre de réponses possibles (en réalité, une seule réponse possible pour toutes les questions du dataset)


### 4. UltraMedical-Preference
Lien :  https://huggingface.co/datasets/TsinghuaC3I/UltraMedical-Preference

#### Nature du dataset
UltraMedical-Preference est un dataset de préférences pour l’alignement. Le projet UltraMedical annonce une collection avec plus de 100k données de préférence au sein d’un ensemble biomédical plus large.

#### Apport métier
UltraMedical-Preference n’est ni un dataset de QA ouverte ni un dataset de QCM. C’est un dataset pour préférence / ranking / DPO.

#### Champs

* `prompt` : Requête / consigne
* `chosen` : réponse préférée, stockée comme liste de messages avec content (texte complet)
* `rejected` : réponse inadaptée, au même format
* `feedback` : justification textuelle du choix
* `prompt_id` : identifiant de la question
* `label_type` : Type de comparaison (sur quoi se base la préférence choisie, ex : sureté, la plus factuellement correcte, etc.)
* `metadata` : métadonnées encodées

### Aggrégation : Construction d'une structure commune

Afin de réaliser l'entraînement, il est nécessaire d'avoir un dataset homogène.
Nous allons donc utiliser une structure commune.

Voici la structure qui a été retenue : 

```
{
  "id": "mediqal-oeq-00000001",
  "dataset": "mediqal",
  "language": "fr",
  "instruction": "Réponds de manière claire, concise et médicale à la question suivante.",
  "input": "Cas clinique : ...\n\nQuestion : ...",
  "output": "Réponse attendue...",
  "metadata": {
    "task_type": "qa_open",
    "medical_subject": "cardiologie",
    "question_type": "reasoning",
    "has_clinical_case": true,
    "source_row_id": "12345"
  }
}
```
## Entrainement

Pipeline:
1. Preparation des jeux de données
2. Entrainement SFT
3. Entrainement DPO à partir du modèle SFT

Commandes minimales (dans le conteneur `p14-train`):

```bash
python -m src.dataset.main
python -m src.train.sft
python -m src.train.dpo
```

Optionnel: demarrer DPO depuis un run SFT W&B (au lieu du local):

```bash
SFT_WANDB_RUN_PATH=<entity>/<project>/<run_id> python -m src.train.dpo
```

Notes:
- Toujours local: le SFT ecrit l'adapter final + checkpoints dans `artifacts/sft`.
- Optionnel W&B: les checkpoints/modèles sont aussi uploadés vers W&B si W&B est actif.
- Le mode d'upload est contrôlé par `WANDB_LOG_MODEL`:
  - `checkpoint`: upload des checkpoints (et modèle final)
  - `end`: upload du modèle final seulement
- Si `SFT_WANDB_RUN_PATH` n'est pas renseigné, DPO charge l'adapter local (`artifacts/sft`) par défaut.

## Evaluation clinique

Le jeu `clinical_eval.jsonl` est un holdout separe (pas utilise pour l'entrainement).

- `python -m src.train.sft` lance une evaluation clinique en fin de run.
- `python -m src.train.dpo` lance aussi une evaluation clinique en fin de run.
- Les metriques sont logguees dans W&B avec le prefixe `clinical_eval/`.
- Les resultats sont aussi ecrits dans `artifacts/sft/eval_report.json` et `artifacts/dpo/eval_report.json`.

Variables utiles (optionnelles):
- `CLINICAL_EVAL_ENABLED` (defaut: `1`)
- `CLINICAL_EVAL_MAX_LENGTH` (defaut: max length du run)
- `CLINICAL_EVAL_BATCH_SIZE` (defaut: batch size du run)
