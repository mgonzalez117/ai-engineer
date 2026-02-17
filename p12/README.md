# Projet 12 -- Pipeline d'extraction multimodale

## 1. Objectif

Ce projet met en place une architecture conteneurisée permettant :

-   l'extraction automatisée de données multimodales (texte + image),
-   leur transformation et normalisation,
-   leur chargement en base PostgreSQL,
-   l'orchestration avec Airflow,
-   la visualisation des indicateurs via Streamlit.

L'environnement est entièrement géré par Docker Compose afin d'assurer
reproductibilité et portabilité.

------------------------------------------------------------------------

## 2. Architecture

Services définis dans `docker-compose.yml` :

-   p12-notebooks\
-   p12-extract-transform\
-   p12-datastore\
-   p12-airflow\
-   p12-pgadmin\
-   p12-dashboard

Tous les services communiquent via le réseau Docker :

    p12_network

------------------------------------------------------------------------

## 3. Description des services

### p12-notebooks

Environnement de développement interactif (Jupyter Lab).

Port : http://localhost:8888

Permet : - exploration des datasets, - tests d'API, - prototypage des
scripts ETL.

------------------------------------------------------------------------

### p12-extract-transform

Exécute les scripts d'extraction et de transformation :

    python -m src.1-extract.fakeddit.main &&
    python -m src.2-transform.fakeddit.pipeline_transform &&
    python -m src.3-load

Responsabilités : 1. Extraction texte + image 2. Nettoyage et
normalisation 3. Vérification cohérence texte/image 4. Export des
données transformées

Variables utilisées : - DATA_RAW_DIR - DATA_PROCESSED_DIR -
DATA_FINAL_DIR

------------------------------------------------------------------------

### p12-datastore

Base PostgreSQL (image : postgres:13).

Port : localhost:5433

Persistance via volume : datastore-data.

------------------------------------------------------------------------

### p12-airflow

Orchestration du pipeline ETL.

Port : http://localhost:8080

Fonctions : - gestion du DAG ETL, - séparation Extract / Transform /
Load, - journalisation des exécutions.

------------------------------------------------------------------------

### p12-pgadmin

Interface d'administration PostgreSQL.

Port : http://localhost:5050

Permet d'inspecter les tables et valider les insertions.

------------------------------------------------------------------------

### p12-dashboard

Dashboard Streamlit (Dockerfile.streamlit).

Port : http://localhost:8501

Affiche les KPI du pipeline : - nombre de publications extraites, - taux
d'images valides, - temps d'exécution, - répartition des labels.

------------------------------------------------------------------------

## 4. Lancement

Depuis la racine du projet :

    docker compose up --build

En arrière-plan :

    docker compose up -d --build

Arrêt :

    docker compose down

------------------------------------------------------------------------

## 5. Données d'origine

Les données principales provenant des datasets choisis (notamment Fakeeddit) sont téléchargées dans `data/raw` lors du premier lancement.
Elles ne sont pas retéléchargées si déjà présentes.

------------------------------------------------------------------------

## 6. Structure du projet

    src/
     ├── 0-notebooks/   # notebooks pour l'exploration des datasets 
     ├── 1-extract/     # phase d'extraction
     ├── 2-transform/   # phase de transformation
     ├── 2-load/        # phase de load
     ├── dashboard/     # code avec app.py pour le dashboard
     ├── models/        # modèles SQLAlchemy pour intéragir avec la base de données



    airflow/    # pipelines airflow
    postgres/   # SQL d'initialisation de la base de données
    data/       # données de travail

------------------------------------------------------------------------

## 7. Bilan

Cette architecture permet :

-   extraction automatisée multimodale,
-   pipeline reproductible,
-   orchestration Airflow,
-   stockage SQL,
-   visualisation et monitoring des performances.
