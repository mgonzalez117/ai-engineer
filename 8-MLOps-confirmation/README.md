---
title: Projet 8 - MLOps part 2
emoji: ⚡
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
---

# Confirmez vos compétences en MLOps (2/2)

## Comment utiliser ce livrable

### Makefile

Le fichier Makefile permet de déployer la solution complète en peu de temps.

Assurez-vous d'avoir make installé sur votre hôte local : 
```
sudo apt-get install make
```

Lancez le makefile depuis cette commande : 
```
make start
```

Ce que fait le Makefile : 

* Copie `.env.dist` vers `.env` (/!\ **assurez-vous d'avoir renseigné les variables dans .env**)
* Télécharge les **données d'entraînement** si elles n'existent pas
* Construit et lance la stack **docker** en local
* Exécute les scripts suivants : 
  * (`scripts/extract_single`) Extraction d'un extrait des données test et bureau en :
    * .json pour être utilisé dans le test d'API
    * .csv pour être utilisé par le profiling du code de prediction
  * (`scripts/profile/predict`) Lancement du script de profiling 
    * Profile le code en simulant un /predict
    * Extrait le résultat dans un fichier `.prof` dans `/.data/profiling/`
* Construit et lance le container `p8-snakeviz` permettant de visualiser le résultat du profiling à l'adresse suivante : 
  * http://localhost:8082/snakeviz/%2Fapp%2Fprofiling%2Fpredict_inference.prof (**port à adapter à votre configuration**)

#### Données d'entraînement
Les données d'entrainement du modèle sont disponibles à cette adresse : https://s3-eu-west-1.amazonaws.com/static.oc-static.com/prod/courses/files/Parcours_data_scientist/Projet+-+Impl%C3%A9menter+un+mod%C3%A8le+de+scoring/Projet+Mise+en+prod+-+home-credit-default-risk.zip.
Elles sont automatiquement téléchargées par le Makefile

le dossier n'est pas versionné car la taille des données est importante (~2,5G).

### Docker

L'environnement mis à disposition fonctionne avec docker. 
Il est nécessaire d'avoir docker, ainsi que docker-compose installé sur votre machine hôte pour le lancer.

* Copiez le fichier `.env.dist` vers  `.env`
* Lancez la stack docker (avec `docker-compose up -d`, par exemple, ou directement via le Makefile).

## Modèle de scoring

Pour pouvoir réutiliser le modèle de scoring, il faut permettre : 

* Le traitement des données d'entrée tel que le modèle entrainé
* Le chargement du modèle pré-entraîné

###  Hébergement du modèle

Le modèle et la pipeline de pré-traitement des données sont stockés sur Hugging Faces model à l'adresse suivante : https://huggingface.co/MGonzalez117/home-credit

### Comment refaire un build

Le Makefile ne fait pas le build du modèle car ce dernier a déjà été construit et stocké sur Hugging Faces.
Si vous souhaitez modifier et reconstruire le modèle, nous avons créé un script de build : `src/ml/build.py`, ce script : 

* Nettoie, normalise et encode les données d'entrainement dans `.data/modelisation/app_train.csv`
  * Enregistre les traitements dans une **pipeline réutilisable ultérieurement via l'API** : `models/pipeline.joblib`
* Entraîne le modèle depuis les données d'entrainement
  * Enregistre le modèle pré entrainé pour être **réutilisable ultérieurement via l'API** : `models/model.joblib`

**/!\ Ensuite, Mettre à jour le dépôt HuggingFace avec le nouveau modèle généré** : 

```
cd models/
git add *.joblib
git commit -m "Model & pipeline Update" # adapter le message de commit
git push origin main
```

Remarque : le dossier `models/` est configuré sur un autre dépôt Git que le projet, afin d'héberger le modèle : https://huggingface.co/MGonzalez117/home-credit

## Tests et déploiement de l'API 

### Gitlab CI/CD

On utilise la CI/CD Gitlab avec le fichier de configuration `.gitlab-ci.yml`, dont voici le workflow : 

* feature, develop, release:
   * build image
   * test
*  main :
   * build image
   * test
   * déploiement sur hugging faces spaces (si tests réussis)

#### Variables & secrets Gitlab

Il est nécessaire de configurer les variables sur Gitlab, et de les protéger : 

* `HF_SPACE_REPOSITORY` : pour permettre l'identification du repository Hugging Face Spaces
  * valeur disponible dans `.env.dist` : https://huggingface.co/spaces/MGonzalez117/ai-engineer-p8
* `HF_SPACE_TOKEN` : pour permettre le déploiement sur ce repository
  * secret, demander à l'administrateur de l'espace Hugging Face 

### Hugging Face space

L'API est hébergée en production sur l'espace Hugging Faces suivant : https://huggingface.co/spaces/MGonzalez117/ai-engineer-p8

#### Variables & secrets HuggingFaces

Il est nécessaire de préconfigurer cet environnement avec les variables suivantes (déjà réalisé pour ce projet) : 

* `DATABASE_URL`, avec pour valeur `sqlite:////tmp/db.sqlite` 
* `API_TOKEN`, avec votre secret (token pour accéder à l'API)
* `HF_REPOSITORY` : pré-rempli dans `.env.dist`
* `HF_MODEL` : pré-rempli dans `.env.dist`
* `HF_PIPELINE` : pré-rempli dans `.env.dist`
* `XDG_CACHE_HOME` : `/tmp/.cache`

## Détection automatique du Datadrift

Le code permettant de gérer le rapport HTML du datadrift est inclus dans `src/drift/monitoring.py`
Voici son fonctionnement : 

* Récupère les données de logs de prédictions depuis la base de données (`predict_logs`)
* Compare ces données de production au jeu d'entrainement local avec **Evidently**
* Génère le rapport au format HTML dans `.data/drift/report.html`

Ce script de monitoring s'exécute en production comme en local une fois par mois
La configuration est faite dans le `Dockerfile`, par l'utilisation de `crontab`

## Profiling automatique du code de prédiction du modèle


Le profiling du code s'effectue automatiquement, il lancé lors de la création de la stack locale avec `make start` (il n'est pas utile de l'exporter en production), il permet : 
* de simuler un appel à `/predict` et analyse la stack d'appels avec `cProfile`
* de générer un rapport d'analyse dans `.data/profiling/`, en html et en `.prof`
* de lancer le container docker `p8-snakeviz` pour visualiser les résultats du rapports à l'adresse suivante : 
  * http://localhost:8082/snakeviz/%2Fapp%2Fprofiling%2Fpredict_inference.prof (**port à adapter à votre configuration**)

### Premières optimisations 

Nous avons réalisé un profiling détaillé de la fonction predict_inference (100 prédictions).
Le temps total était de 21,1 s.


#### Goulots d’étranglement identifiés :

* Téléchargement du modèle à chaque appel
* Prétraitement lourd à base de nombreuses opérations Pandas colonne par colonne.


