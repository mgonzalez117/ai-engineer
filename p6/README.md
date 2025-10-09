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

