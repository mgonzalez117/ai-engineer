# Plan de Monitoring – Pipeline ETL Multi-Sources

## 1. Objectif

Ce document décrit la stratégie de supervision du pipeline ETL conçu pour ingérer et traiter plusieurs sources de données (ex. Fakeddit et autres datasets labellisés pour les FakeNews).

Le pipeline suit une architecture standard :

- Extract  
- Transform  
- Load  

L’objectif du monitoring est de garantir :

- la fiabilité d’exécution  
- la qualité des données  
- la performance du traitement  
- la traçabilité des runs

---

## 2. Architecture de Supervision

La supervision repose sur deux couches complémentaires :

### 2.1 Monitoring d’orchestration – Airflow

Airflow assure le suivi technique du pipeline :

- Statut des DAG runs  
- Statut des tâches (extract, transform, load)  
- Durée des tâches  
- Nombre de retries  
- Logs détaillés  

Airflow est la source de vérité concernant l’état d’exécution.

---

### 2.2 Monitoring Data – Table `etl_metrics`

Une table dédiée `etl_metrics` centralise les indicateurs de chaque étape et de chaque source.

Chaque enregistrement contient :

- `pipeline_name`
- `step`
- `run_id`
- `nb_input`
- `nb_output`
- `nb_rejected`
- `duration_seconds`
- `rows_per_second`
- `success`
- `created_at`

Cette séparation permet de distinguer :

- Les données métier (`publication`)
- Les métriques de supervision (`etl_metrics`)

Les métriques sont visualisées via un dashboard Streamlit.

---

## 3. Indicateurs Surveillés

Les KPI sont définis de manière générique afin d’être applicables à toute source intégrée.

### 3.1 Volumétrie

- **nb_input** : volume en entrée  
- **nb_output** : volume produit  
- **nb_rejected** : volume rejeté  
- **Taux de validité** = nb_output / nb_input  
- **Taux de rejet** = nb_rejected / nb_input  

Objectifs :

- Détecter une chute brutale de volume  
- Identifier une anomalie spécifique à une source  
- Comparer les comportements entre sources  

---

### 3.2 Performance

- **duration_seconds**
- **rows_per_second**
- Moyenne des durées

Objectifs :

- Identifier un ralentissement progressif  
- Détecter une dégradation liée à une nouvelle source  
- Vérifier la scalabilité du pipeline  

---

### 3.3 Fiabilité

- Taux de succès des runs  
- Répétition d’échecs sur une même étape  
- Échec spécifique à une source  

---

## 4. Seuils d’Alerte

Les seuils suivants sont définis à titre indicatif :

| Indicateur                                 | Condition | Action |
|--------------------------------------------|-----------|--------|
| nb_output = 0                              | Aucune donnée produite | Analyse immédiate |
| Taux de validité < 98%                     | Rejets anormalement élevés | Vérification transform |
| duration_seconds > 1,5× moyenne historique | Dégradation performance | Vérifier base / batch |
| 2 runs consécutifs en échec                | Instabilité | Analyse technique |

Ces seuils peuvent être ajustés en fonction des volumes propres à chaque source.

---

## 5. Fréquence de Vérification

- Vérification des runs Airflow : à chaque exécution  
- Consultation du dashboard KPI : hebdomadaire  
- Analyse des tendances performance : mensuelle  
- Révision des seuils : lors de l’intégration d’une nouvelle source  

---

## 6. Procédures en Cas d’Incident

### 6.1 Échec d’Extract

- Vérifier la disponibilité de la source  
- Vérifier la structure du fichier d’entrée  
- Vérifier la connectivité réseau  

### 6.2 Échec de Transform

- Vérifier l’évolution du schéma source  
- Analyser les fichiers rejetés  
- Vérifier la compatibilité avec le modèle cible  

### 6.3 Échec de Load

- Vérifier la connexion à la base de données  
- Vérifier les contraintes d’intégrité  
- Vérifier la taille des batchs  

### 6.4 Dégradation de Performance

- Vérifier les index en base  
- Vérifier les ressources du conteneur  
- Analyser la volumétrie par source  

---

## 7. Extensibilité Multi-Sources

L’architecture de monitoring est conçue pour supporter l’ajout de nouvelles sources sans modification structurelle :

- Les métriques sont homogènes entre sources  
- Chaque source est identifiable via `pipeline_name`  
- Le dashboard permet le filtrage par pipeline et par étape  

Cela permet :

- La comparaison des performances entre sources  
- L’identification rapide d’une source problématique  
- Le suivi de la montée en charge globale du pipeline  

---

## 8. Conclusion

La stratégie de monitoring repose sur :

- Airflow pour la supervision d’exécution  
- Une table dédiée `etl_metrics` pour la supervision data  
- Un dashboard unifié pour la visualisation des KPI  

Cette approche respecte les bonnes pratiques d’ingénierie des données en séparant orchestration et observabilité, tout en garantissant l’évolutivité du pipeline.
