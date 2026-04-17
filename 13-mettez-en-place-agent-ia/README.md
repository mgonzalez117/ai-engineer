# ♟ AI Engineer -- Projet 13

## Agent IA d'apprentissage des ouvertures (FFE)

Ce document décrit exclusivement l'architecture Docker du projet et le
rôle précis de chaque conteneur.

Le système repose sur une architecture micro-services orchestrée avec
Docker Compose.

------------------------------------------------------------------------

# Architecture globale

Nom du projet Docker :

name: ai-engineer-p13

Tous les services communiquent via un réseau interne dédié :

networks: p13_network: driver: bridge

------------------------------------------------------------------------

# Services Docker

## 1. p13-api --- Backend FastAPI

Conteneur principal contenant l'API de l'agent IA.

### Rôle

-   Expose les endpoints métier :
    -   /api/v1/moves/{fen}
    -   /api/v1/evaluate/{fen}
    -   /vector-search
    -   /api/v1/videos/{opening}
-   Orchestre les appels vers :
    -   Milvus (recherche vectorielle)
    -   Lichess API
    -   Stockfish
    -   YouTube API

### Configuration principale

p13-api: container_name: p13-api build: . ports: - "8000:8000"

Accessible via : http://localhost:8000

------------------------------------------------------------------------

## 2. p13-milvus --- Base de données vectorielle

Milvus est la base vectorielle utilisée pour implémenter le RAG
(Retrieval-Augmented Generation).

### Rôle

-   Stocke les embeddings générés à partir des données Wikichess
-   Permet la recherche sémantique par similarité vectorielle
-   Retourne les passages les plus pertinents pour enrichir les réponses
    de l'agent

### Ports exposés

-   19530 : gRPC (connexion backend)
-   9091 : HTTP

### Volume persistant

./.data/milvus:/var/lib/milvus

------------------------------------------------------------------------

## 3. p13-etcd --- Stockage des métadonnées Milvus

etcd est une base clé-valeur distribuée.

### Rôle dans l'architecture

Milvus ne fonctionne pas seul. Il a besoin d'un système externe pour
stocker : - Ses métadonnées - La configuration des collections - Les
informations de cluster - Les états internes

etcd joue ce rôle de registre interne pour Milvus.

Sans etcd : - Milvus ne peut pas démarrer correctement - Les collections
vectorielles ne sont pas persistées correctement

### Volume

./.data/etcd:/etcd

------------------------------------------------------------------------

## 4. p13-minio --- Stockage objet compatible S3

MinIO est un serveur de stockage objet compatible Amazon S3.

### Rôle dans l'architecture

Milvus utilise MinIO pour stocker : - Les fichiers volumineux liés aux
index vectoriels - Les segments de données - Les fichiers binaires
internes

Pourquoi MinIO ?

Milvus sépare : - Les métadonnées (etcd) - Les données volumineuses
(stockage objet)

MinIO joue donc le rôle de backend de stockage pour les données
physiques.

### Ports exposés

-   9000 : API S3
-   9001 : Console Web

Identifiants (environnement local uniquement) :
MINIO_ROOT_USER=minioadmin\
MINIO_ROOT_PASSWORD=minioadmin

Console : http://localhost:9001

------------------------------------------------------------------------

# Dépendances et ordre de démarrage

Ordre logique :

p13-etcd\
p13-minio\
→ p13-milvus\
→ p13-api

Milvus dépend obligatoirement de etcd et MinIO.

------------------------------------------------------------------------

# Persistance des données

Les données sont stockées localement dans :

.data/ ├── etcd/ ├── minio/ └── milvus/

Cela permet : - La persistance après redémarrage - La reconstruction des
conteneurs sans perte de données - La reproductibilité du POC

------------------------------------------------------------------------

# Lancement du projet

docker compose up --build

------------------------------------------------------------------------

# Vérification

docker ps\
http://localhost:8000/docs

------------------------------------------------------------------------

# Résumé technique

-   Backend applicatif : FastAPI
-   Base vectorielle : Milvus
-   Métadonnées Milvus : etcd
-   Stockage objet Milvus : MinIO
-   Réseau interne dédié : bridge Docker
-   Données persistantes via volumes locaux
