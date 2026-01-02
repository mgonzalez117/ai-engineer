# Projet 6 - MLOps (part I)

## Comment utiliser ce livrable

### Pré-requis

* `make`
* `docker`
* `docker-compose`

### Make

Pour lancer la stack, il vous suffit d'exécuter la commande suivante : 
```
make install
```

Cette commande : 

* Télécharge et extrait les images de radiographies dans `.data/`
* Lance la stack `docker`

### Docker

J'ai choisi de gérer les environnements directement avec docker pour une facilité 
de partage, déploiement et une meilleure lisibilité.

Vous retrouverez à la racine du projet le fichier `docker-compose.yaml` vous permettant
de lancer le container pour ce projet. une fois le container lancé vous pouvez accéder aux outils à l'adresse suivante : 

* Jupyter : `127.0.0.1:8888`
* MLFlow : `127.0.0.1:5000`