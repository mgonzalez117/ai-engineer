import pytest

# Payload de base valide pour tous les tests
@pytest.fixture
def base_valid_payload():
    return {
        "age": 35,
        "genre": "M",
        "revenu_mensuel": 3500.0,
        "nombre_experiences_precedentes": 2,
        "annee_experience_totale": 10,
        "annees_dans_l_entreprise": 5,
        "annees_dans_le_poste_actuel": 3,
        "note_evaluation_precedente": 4,
        "niveau_hierarchique_poste": 2,
        "note_evaluation_actuelle": 4,
        "nombre_participation_pee": 2,
        "nb_formations_suivies": 5,
        "distance_domicile_travail": 15,
        "niveau_education": 3,
        "annees_depuis_la_derniere_promotion": 2,
        "annes_sous_responsable_actuel": 3,
        "satisfaction_employee_environnement": 4.0,
        "satisfaction_employee_nature_travail": 3.5,
        "satisfaction_employee_equipe": 4.0,
        "satisfaction_employee_equilibre_pro_perso": 2.5,
        "statut_marital": "Marié(e)",
        "departement": "Consulting",
        "poste": "Manager",
        "domaine_etude": "Infra & Cloud",
        "heure_supplementaires": "Oui",
        "augementation_salaire_precedente": "5%",
        "frequence_deplacement": "Occasionnel"
    }