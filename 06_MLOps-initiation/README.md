# Projet 6 - MLOps (part I)

## Comment utiliser ce livrable

### Pré-requis : Données
Avant de lancer l'environnement il est nécessaire de pré-télécharger les données à cette adresse : https://s3-eu-west-1.amazonaws.com/static.oc-static.com/prod/courses/files/Parcours_data_scientist/Projet+-+Impl%C3%A9menter+un+mod%C3%A8le+de+scoring/Projet+Mise+en+prod+-+home-credit-default-risk.zip
Ensuite, placez-les dans le dossier `.data/`.

Ce dernier n'est pas versionné car la taille des données est importante (~2,5G).

### Docker

J'ai choisi de gérer les environnements directement avec docker pour une facilité 
de partage, déploiement et une meilleure lisibilité.

Vous retrouverez à la racine du projet le fichier `docker-compose.yaml` vous permettant
de lancer le container pour ce projet. une fois le container lancé vous pouvez accéder à MLFlow à l'adresse suivante : 

* MLFlow : `127.0.0.1:5000`

### Modélisation

# Pré-requis : Analyse & Préparation

J'ai réutilisé en partie le kernel `https://www.kaggle.com/code/willkoehrsen/start-here-a-gentle-introduction/notebook`, il comprend : 
* une analyse des données
* le nettoyage
* l'encodage et la normalisation
* quelques features (car le jeu de données `application_test.csv` est déjà en partie un regroupement de features)

**Il est nécessaire d'exécuter ce notebook**, un nouveau fichier résultant est créé pour la modélisation, il nous servira tout au long de nos expérimentations : 
* `./data/modelisation/app_train.csv`

# Expérimentations

Utiliser le notebbok `src/2-experimentations`, il permet de lancer les essais sur MLFlow localement.