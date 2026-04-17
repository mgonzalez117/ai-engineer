import pandas as pd
import logging
from sqlalchemy import create_engine, text
from pathlib import Path
from src.models.base import Base
from src.database import get_db_engine, is_database_empty

BASE_DIR = Path(__file__).resolve().parent.parent.parent
ASSETS_DIR = BASE_DIR / "assets"
SQL_DIR = BASE_DIR / "sql"

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def init_schema(engine):
    """Exécute le script d'init SQL"""
    with engine.begin() as conn:
        with open(SQL_DIR / "init.sql", "r") as f:
            sql_script = f.read()
        # SQLite exécute tout le script d’un coup
        for stmt in sql_script.split(";"):
            if stmt.strip():
                conn.execute(text(stmt))

def create_orm_tables(engine):
    Base.metadata.create_all(bind=engine)

def execute_ignore_conflicts(conn, sql, params=None):
    """Exécute une requête en ignorant les conflits"""
    try:
        if params:
            conn.execute(text(sql), params)
        else:
            conn.execute(text(sql))
    except Exception as e:
        # Ignore les erreurs de contrainte (doublons)
        if "UNIQUE constraint failed" in str(e) or "duplicate key" in str(e) or "already exists" in str(e):
            pass  # Ignore les doublons
        else:
            raise  # Re-raise les autres erreurs

def populate_reference_tables(engine):
    """Peuple les tables de référence"""
    with engine.begin() as conn:
        # Domaines d'étude
        domaines = [
            (1, 'Infra & Cloud'),
            (2, 'Transformation Digitale'),
            (3, 'Marketing'),
            (4, 'Ressources Humaines'),
            (5, 'Entrepreunariat'),
            (6, 'Autre')
        ]

        for domaine_id, intitule in domaines:
            execute_ignore_conflicts(conn,
                                     "INSERT INTO sirh_domaine_etude (domaine_etude_id, intitule) VALUES (:id, :nom)",
                                     {"id": domaine_id, "nom": intitule}
                                     )

        # Départements
        departements = [
            (1, 'Commercial'),
            (2, 'Consulting'),
            (3, 'Ressources Humaines')
        ]

        for dept_id, nom in departements:
            execute_ignore_conflicts(conn,
                                     "INSERT INTO sirh_departement (departement_id, nom) VALUES (:id, :nom)",
                                     {"id": dept_id, "nom": nom}
                                     )

        # Postes
        postes = [
            (1, 'Cadre Commercial'),
            (2, 'Assistant de Direction'),
            (3, 'Consultant'),
            (4, 'Tech Lead'),
            (5, 'Manager'),
            (6, 'Représentant Commercial'),
            (7, 'Ressources Humaines'),
            (8, 'Senior Manager'),
            (9, 'Directeur Technique')
        ]

        for poste_id, intitule in postes:
            execute_ignore_conflicts(conn,
                                     "INSERT INTO sirh_poste (poste_id, intitule) VALUES (:id, :nom)",
                                     {"id": poste_id, "nom": intitule}
                                     )

        # Statuts maritaux
        statuts = [
            (1, 'Célibataire'),
            (2, 'Marié(e)'),
            (3, 'Divorcé(e)')
        ]

        for statut_id, intitule in statuts:
            execute_ignore_conflicts(conn,
                                     "INSERT INTO sirh_statut_marital (statut_marital_id, intitule) VALUES (:id, :nom)",
                                     {"id": statut_id, "nom": intitule}
                                     )

        # Fréquences de déplacement
        frequences = [
            (1, 'Occasionnel'),
            (2, 'Frequent'),
            (3, 'Jamais')
        ]

        for freq_id, intitule in frequences:
            execute_ignore_conflicts(conn,
                                     "INSERT INTO sirh_frequence_deplacement (frequence_deplacement_id, intitule) VALUES (:id, :nom)",
                                     {"id": freq_id, "nom": intitule}
                                     )

        logger.info("Tables de référence peuplées avec succès")


def create_mappings():
    """Crée les mappings pour les clés étrangères"""
    return {
        'domaine_etude': {
            'Infra & Cloud': 1,
            'Transformation Digitale': 2,
            'Marketing': 3,
            'Ressources Humaines': 4,
            'Entrepreunariat': 5,
            'Autre': 6
        },
        'departement': {
            'Commercial': 1,
            'Consulting': 2,
            'Ressources Humaines': 3
        },
        'poste': {
            'Cadre Commercial': 1,
            'Assistant de Direction': 2,
            'Consultant': 3,
            'Tech Lead': 4,
            'Manager': 5,
            'Représentant Commercial': 6,
            'Ressources Humaines': 7,
            'Senior Manager': 8,
            'Directeur Technique': 9
        },
        'statut_marital': {
            'Célibataire': 1,
            'Marié(e)': 2,
            'Divorcé(e)': 3
        },
        'frequence_deplacement': {
            'Occasionnel': 1,
            'Frequent': 2,
            'Aucun': 3
        }
    }


def load_and_merge_data():
    """Charge et fusionne les données des fichiers CSV"""
    df_sirh = pd.read_csv( ASSETS_DIR / 'extrait_sirh.csv', delimiter=',')
    df_eval = pd.read_csv( ASSETS_DIR / 'extrait_eval.csv', delimiter=',')
    df_sondage = pd.read_csv( ASSETS_DIR / 'extrait_sondage.csv', delimiter=',')

    df_eval['eval_number'] = df_eval['eval_number'].str.replace("E_", "").astype(int)
    df_sondage['code_sondage'] = df_sondage['code_sondage'].astype(int)

    df_merged = df_sirh.merge(df_eval, left_on='id_employee', right_on='eval_number', how='left')
    df_merged = df_merged.merge(df_sondage, left_on='id_employee', right_on='code_sondage', how='left')

    return df_merged


def populate_employees(engine, df, mappings):
    """Peuple la table sirh_employee"""
    with engine.begin() as conn:
        for _, row in df.iterrows():
            a_quitte = row['a_quitte_l_entreprise'] == 'Oui'
            heures_supp = row.get('heure_supplementaires', 'Non') == 'Oui'

            employee_data = {
                'employee_id': int(row['id_employee']),
                'a_quitte_l_entreprise': a_quitte,
                'age': int(row['age']),
                'genre': row['genre'],
                'revenu_mensuel': int(row['revenu_mensuel']),
                'nombre_experiences_precedentes': int(row['nombre_experiences_precedentes']),
                'annees_experience_totale': int(row['annee_experience_totale']),
                'annees_dans_l_entreprise': int(row['annees_dans_l_entreprise']),
                'annees_dans_le_poste_actuel': int(row['annees_dans_le_poste_actuel']),
                'niveau_education': int(row.get('niveau_education')),
                'heures_supplementaires': heures_supp,
                'statut_marital_id': mappings['statut_marital'].get(row['statut_marital']),
                'departement_id': mappings['departement'].get(row['departement']),
                'poste_id': mappings['poste'].get(row['poste']),
                'domaine_etude_id': mappings['domaine_etude'].get(row.get('domaine_etude', 'Autre'))
            }

            execute_ignore_conflicts(conn, """
                                           INSERT INTO sirh_employee (employee_id, a_quitte_l_entreprise, age, genre,
                                                                      revenu_mensuel,
                                                                      nombre_experiences_precedentes,
                                                                      annees_experience_totale,
                                                                      annees_dans_l_entreprise,
                                                                      annees_dans_le_poste_actuel,
                                                                      niveau_education, heures_supplementaires,
                                                                      statut_marital_id,
                                                                      departement_id, poste_id, domaine_etude_id)
                                           VALUES (:employee_id, :a_quitte_l_entreprise, :age, :genre, :revenu_mensuel,
                                                   :nombre_experiences_precedentes, :annees_experience_totale,
                                                   :annees_dans_l_entreprise, :annees_dans_le_poste_actuel,
                                                   :niveau_education, :heures_supplementaires, :statut_marital_id,
                                                   :departement_id, :poste_id, :domaine_etude_id)
                                           """, employee_data)

    logger.info(f"{len(df)} employés traités")


def populate_evaluations(engine, df):
    """Peuple la table sirh_evaluation"""
    with engine.begin() as conn:
        count = 0
        for _, row in df.iterrows():
            if pd.notna(row.get('eval_number')):
                eval_data = {
                    'evaluation_id': int(row['eval_number']),
                    'satisfaction_employee_environnement': int(row.get('satisfaction_employee_environnement')),
                    'note_evaluation_precedente': int(row.get('note_evaluation_precedente')),
                    'niveau_hierarchique_poste': int(row.get('niveau_hierarchique_poste')),
                    'satisfaction_employee_nature_travail': int(row.get('satisfaction_employee_nature_travail')),
                    'satisfaction_employee_equipe': int(row.get('satisfaction_employee_equipe')),
                    'satisfaction_employee_equilibre_pro_perso': int(
                        row.get('satisfaction_employee_equilibre_pro_perso')),
                    'note_evaluation_actuelle': int(row.get('note_evaluation_actuelle')),
                    'augementation_salaire_precedente': float(
                        str(row.get('augementation_salaire_precedente', '0')).replace('%', '').replace(',',
                                                                                                       '.').strip()) / 100,
                    'employee_id': int(row['id_employee'])
                }

                execute_ignore_conflicts(conn, """
                                               INSERT INTO sirh_evaluation (evaluation_id,
                                                                            satisfaction_employee_environnement,
                                                                            note_evaluation_precedente,
                                                                            niveau_hierarchique_poste,
                                                                            satisfaction_employee_nature_travail,
                                                                            satisfaction_employee_equipe,
                                                                            satisfaction_employee_equilibre_pro_perso,
                                                                            note_evaluation_actuelle,
                                                                            augementation_salaire_precedente,
                                                                            employee_id)
                                               VALUES (:evaluation_id, :satisfaction_employee_environnement,
                                                       :note_evaluation_precedente,
                                                       :niveau_hierarchique_poste,
                                                       :satisfaction_employee_nature_travail,
                                                       :satisfaction_employee_equipe,
                                                       :satisfaction_employee_equilibre_pro_perso,
                                                       :note_evaluation_actuelle, :augementation_salaire_precedente,
                                                       :employee_id)
                                               """, eval_data)
                count += 1

        logger.info(f"{count} évaluations insérées")


def populate_sondages(engine, df, mappings):
    """Peuple la table sirh_sondage"""
    with engine.begin() as conn:
        count = 0
        for _, row in df.iterrows():
            if pd.notna(row.get('code_sondage')):
                sondage_data = {
                    'sondage_id': int(row['code_sondage']),
                    'nombre_participation_pee': int(row.get('nombre_participation_pee')),
                    'nb_formations_suivies': int(row.get('nb_formations_suivies')),
                    'nombre_employee_sous_responsabilite': int(row.get('nombre_employee_sous_responsabilite')),
                    'distance_domicile_travail': int(row.get('distance_domicile_travail')),
                    'annees_depuis_la_derniere_promotion': int(row.get('annees_depuis_la_derniere_promotion')),
                    'annes_sous_responsable_actuel': int(row.get('annes_sous_responsable_actuel')),
                    'frequence_deplacement_id': mappings['frequence_deplacement'].get(row.get('frequence_deplacement')),
                    'employee_id': int(row['id_employee'])
                }

                execute_ignore_conflicts(conn, """
                                               INSERT INTO sirh_sondage (sondage_id, nombre_participation_pee,
                                                                         nb_formations_suivies,
                                                                         nombre_employee_sous_responsabilite,
                                                                         distance_domicile_travail,
                                                                         annees_depuis_la_derniere_promotion,
                                                                         annes_sous_responsable_actuel,
                                                                         frequence_deplacement_id, employee_id)
                                               VALUES (:sondage_id, :nombre_participation_pee, :nb_formations_suivies,
                                                       :nombre_employee_sous_responsabilite, :distance_domicile_travail,
                                                       :annees_depuis_la_derniere_promotion,
                                                       :annes_sous_responsable_actuel,
                                                       :frequence_deplacement_id, :employee_id)
                                               """, sondage_data)
                count += 1

        logger.info(f"{count} sondages insérés")


def main():
    """Fonction principale d'ingestion des données"""
    try:
        engine = get_db_engine()

        if not is_database_empty(engine):
           logger.info("La base de données contient déjà des données. Arrêt du processus.")
           return

        logger.info("Chargement du schéma depuis sql/init.sql...")
        init_schema(engine)

        create_orm_tables(engine)

        logger.info("Début de l'ingestion...")

        populate_reference_tables(engine)
        mappings = create_mappings()
        df_merged = load_and_merge_data()
        logger.info(f"Données chargées: {len(df_merged)} lignes")

        populate_employees(engine, df_merged, mappings)
        populate_evaluations(engine, df_merged)
        populate_sondages(engine, df_merged, mappings)

        logger.info("Ingestion des données terminée avec succès!")

    except Exception as e:
        logger.error(f"Erreur lors de l'ingestion: {e}")
        raise

main()