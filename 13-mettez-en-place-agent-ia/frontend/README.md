# Projet 13 - Mettez en place un Agent IA

Proof of Concept d'un agent IA pour l'apprentissage des ouvertures aux
échecs réalisé dans le cadre de la formation **AI Engineer --
OpenClassrooms**.

L'application aide les joueurs à apprendre les ouvertures en combinant :

-   les coups théoriques issus de l'API **Lichess**
-   l'évaluation de position via **Stockfish**
-   du contexte provenant de **Wikichess** via une recherche vectorielle
    (RAG)
-   des **vidéos YouTube** explicatives pertinentes

Le système est composé de deux parties principales : 
* un **backend Python (FastAPI)** 
* un **frontend Angular** avec un échiquier interactif.

------------------------------------------------------------------------

# Structure du projet

Le projet est organisé en deux parties principales.

## Backend

Le backend est une application **FastAPI** qui implémente l'agent IA et
ses différents services.

Structure principale :

```text
backend/src 
  ├─ agent/ # Agent IA, implémenté avec LangChain.
  ├─ api/ # Endpoints FastAPI (routes HTTP) pour interfacer l'agent
  ├─ service/ # Logique métier et intégrations externes 
  ├─ rag/ # Ingestion des données et recherche vectorielle
```

### Responsabilités

**Couche API (`api/`)** 
  * Expose les endpoints REST utilisés par le frontend.

**Couche Service (`service/`)** 
  * Contient la logique métier et les intégrations : 
    * API **Lichess** (coups d'ouverture théoriques) 
    * **Stockfish** (évaluation de position) 
    * API **YouTube** (vidéos pédagogiques) 
    * **Milvus** (base vectorielle pour le RAG)

Cette séparation permet de garder une API légère et une logique métier
bien organisée.

------------------------------------------------------------------------

## Frontend

Le frontend est une application **Angular** qui fournit l'interface
utilisateur.

Structure principale :

```text
frontend/ 
    ├─ src/app/
        ├─ chessboard/ # rendu échiquier
        ├─ agent-sidebar/ # barre latérale de l'agent
        ├─ learning-panel/ # barre d'apprentissage : vidéo et contexte
        └─ services/ # Agent et appel api
```

### Fonctionnalités

-   échiquier interactif
-   panneau de recommandations de l'agent affichant :
    -   les coups suggérés
    -   du contexte sur l'ouverture
    -   des vidéos YouTube pertinentes

Le frontend communique avec le backend via des appels HTTP.

------------------------------------------------------------------------

# Première exécution (ingestion RAG)

Avant de lancer l'application pour la première fois, les données
Wikichess doivent être indexées dans **Milvus**.

Depuis le backend, exécuter :

`python -m service.rag.ingest.main`

Ce script :

-   charge les données Wikichess
-   découpe les textes en morceaux (chunking)
-   génère les embeddings
-   stocke les vecteurs dans **Milvus**

Cette étape n'a besoin d'être exécutée **qu'une seule fois**, sauf si la
base vectorielle est réinitialisée.

------------------------------------------------------------------------

# Lancer le serveur de développement

Démarrer le serveur angular en démarrant simplement le container `p13-client`

Puis ouvrir :

http://localhost:3200

L'application se recharge automatiquement lors des modifications du
code.
