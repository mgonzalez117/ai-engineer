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

## Datasets

Nous utilisons les datasets suivants : 

### 1. MediQAl

Lien : https://huggingface.co/datasets/ANR-MALADES/MediQAl

#### Nature du dataset
MediQAl est un dataset français de QA médicale conçu pour évaluer à la fois le rappel factuel et le raisonnement clinique. Il contient 32 603 questions sur 41 sujets médicaux et trois configurations : mcqu, mcqm et oeq. Chaque question est annotée Understanding ou Reasoning

/!\  MediQAl n’est pas un dataset homogène. Il contient en réalité deux natures de tâches :

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
MedQuAD apporte des exemples de QA ouverte en anglais. Il est plus approprié pour du SFT de type open QA, pas du MCQ et pas du preference learning.

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

Afin de réaliser l'entraînement, il est nécessaire de créer une structure commune au dataset qui sera utilisé.
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