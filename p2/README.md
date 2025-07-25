# Projet 2 - Requêtez des services IA

## Comment utiliser ce livrable

### Variables d'environnement

Copiez le fichier `src/.env.dist` vers un fichier `src/.env`, puis complétez les 
variables d'environnement.

```
API_TOKEN=YourApiToken # Token d'Api HuggingFace
IMG_PATH='/workspace/assets/IMG' # dossier contenant le jeu de données d'images
MASKS_PATH='/workspace/assets/Mask' # dossier contenant le jeu de données avec les masques originaux correspondant
BATCH_MAX=3 # maximum d'images à envoyer à l'API lors du batch
```

### Docker

J'ai choisi de gérer les environnements directement avec docker pour une facilité 
de partage, déploiement et une meilleure lisibilité.

* Le Dockerfile cible la version de python `3.13` : `FROM python:3.13-slim`

Vous retrouverez à la racine du projet le fichier `docker-compose.yaml` vous permettant
de lancer le container pour ce projet. une fois le container lancé vous pouvez accéder à Jupyter à l'adresse suivante : 

* JupyterLab URL : `127.0.0.1:8888`

### Poetry

J'ai utilisé Poetry pour gérer les dépendances. Poetry est installé automatiquement depuis le `Dockerfile`.

Vous pouvez relancer Poetry ou installer de nouvelles dépendances en accédant au container docker via : 

* `docker exec -it aiengineer-p2 bash` pour lancer un terminal dans le container
* `poetry add ...` ou `poetry install`, etc.

On peut utiliser les environnements virtuels poetry mais ce n'est pas nécessaire car le container docker
permet déjà l'isolation des couches applicatives.

### Fichiers source et Notebook

Les fichiers source sont exposés dans le répertoire `src/`. 
J'y ai repris le notebook de l'exercice pour le modifier d'après les besoins.

J'ai extrait la plus grande partie du code source dans le dossier `lib/`, vous y trouverez : 

* Le code d'origine fourni dans le notebook
* Le code additionnel pour répondre aux besoins exprimés
