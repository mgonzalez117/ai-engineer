# Projet 3 - Anticipez les besoins en consommations de bâtiments

## Comment utiliser ce livrable

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

* `docker exec -it aiengineer-p3 bash` pour lancer un terminal dans le container
* `poetry add ...` ou `poetry install`, etc.

On peut utiliser les environnements virtuels poetry mais ce n'est pas nécessaire car le container docker
permet déjà l'isolation des couches applicatives.

### Fichiers source et Notebook

Les fichiers source sont exposés dans le répertoire `mission/`. 
J'ai découpé le livrable en 3 notebooks numérotés afin de faciliter la lecture ainsi que la maintenance.