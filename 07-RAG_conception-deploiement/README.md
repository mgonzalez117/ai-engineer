# Projet 7 - Concevoir un système RAG

Ce document permet de comprendre le livrable et de le réutiliser.
Pour plus d'informations, consultez le document technique (PDF) associé au projet.

## Comment utiliser ce livrable

### Docker

L'environnement mis à disposition fonctionne avec docker. 
Il est nécessaire d'avoir docker, ainsi que docker-compose installé sur votre machine hôte pour le lancer.

* Copiez le fichier `.env.dist` vers  `.env`
* Lancez la stack docker (avec `docker-compose up -d`, par exemple).

### Variables d'environnement

* Complétez le fichier `.env` précédemment copié
  * `API_PORT` : le port de votre API qui sera exposée
  * `API_TOKEN` : le token de votre API, vous pouvez utiliser un générateur de mot de passe complexe par exemple
  * `INDEX_DIR` : le dossier dans lequel l'index de base vectorielle est persisté. Privilégiez un volume monté dans docker (`.ragdata\`par exemple)
  * `OPENDATASOFT_URL` : URL de l'API Opendatasoft permettant de récupérer les évènements (utiliser la v2)
  * `FILTER_DEPARTMENT` : Filtre les évènements sur le département donné (cf API)
  * `FILTER_YEARS` : Filtre les évènements sur les années données (cf API)
  * `EMB_MODEL` : modèle qui sera utilisé pour l'embedding. Privilégiez un modèle qui comprend bien le français
  * `PROMPT_FILE` : chemin vers le fichier prompt system, permet de donner les consignes au LLM
  * `MISTRAL_TOKEN` : token API mistral, à générer dans https://admin.mistral.ai/organization/api-keys
  * `DEMO_PORT` : port exposé de la démo du chatbot (Gradio), permet de tester aisément le POC

### Documentation d'API

La documentation d'API est exposée sur votre hôte docker local via le port ${API_PORT}.
* Par défaut http://127.0.0.1:8000

### ChatBot démonstration

Un chatbot de démonstration a été ajouté (container p7-demo), il est acessible sur votre hôte docker local via ${DEMO_PORT}
* Par défaut http://127.0.0.1:8080

## Structure générale

 * `.ragdata/` : volume monté dans docker permettant la persistance de la base vectorielle
 *  `demo/` : dossier dédié au fonctionnement du container de démo (chatbot pour le POC)
 *  `prompts/` : dossier contenant les prompt system pour le LLM
 *  `src/` :
   *  `api/` : exposition de l'Api et des endpoints
   *  `data/`: code dédié au traitement des données (récupération, transformation, découpage, ingestion et persistence de la base)
   *  `service/` : code dédié aux services autour des données (traitement et formulation de la réponse à partir d'une question en langage naturel)
*  `tests/` : dossier dédié pour pytest, voir chapitre suivant
*  `.env.dist` : fichier de configuration de l'environnement, copiez en `.env`
*  `.gitignore` : configuration des dossiers et fichiers à ignorer par GIT
*  `docker-compose.yaml` : orchestration docker pour le POC
*  `Dockerfile` : fichier de build docker pour `p7-api`
*  `poetry.lock`, `pyproject.toml` : configuration de paquets avec poetry
*  `README.md` : ce présent fichier

## Tests

Les tests sont automatisés sur le dépôt github grâce au fichier `.github/workflows/test-p7.yml`

Il est également possible des lancer en local.
Pour ce faire il faut lancer les containers docker puis : 
* Construire l'index en utilisant la requête POST suivante (voir sur doc d'API)
```
curl --location --request PUT 'http://127.0.0.1:8000/rebuild' \
--header 'Authorization: ${API_TOKEN}' \
--data ''
```

Puis se connecter au container `p7-api` et exécuter le test suivant : 
* `pytest ./tests`

**Remarque : Attention l'utilisation de `pytest ./tests/service` peut entrainer l'utilisation de l'API Mistral (pour RAGAS) et donc de la consommation**