# tests/test_prediction_request_simple.py
import pytest
from pydantic import ValidationError
from src.api.prediction_request import PredictionRequest
from tests.fixtures.predict_payload import base_valid_payload

def test_valid_payload_is_accepted(base_valid_payload):
    """Le payload de base doit être accepté sans erreur."""
    request = PredictionRequest(**base_valid_payload)
    assert request is not None

@pytest.mark.parametrize(
    "field, invalid_value, error_msg_part",
    [
        # Tests numériques
        ("age", 17, "age"),
        ("age", 71, "age"),
        ("revenu_mensuel", 0.0, "revenu_mensuel"),
        ("note_evaluation_precedente", 0, "note_evaluation_precedente"),
        ("note_evaluation_actuelle", 6, "note_evaluation_actuelle"),
        ("satisfaction_employee_environnement", 0.9, "satisfaction_employee_environnement"),
        ("satisfaction_employee_equilibre_pro_perso", 5.1, "satisfaction_employee_equilibre_pro_perso"),
        # Tests catégoriels
        ("genre", "X", "genre"),
        ("statut_marital", "Veuf", "statut_marital"),
        ("departement", "Finance", "departement"),
        ("poste", "Stagiaire", "poste"),
        ("domaine_etude", "Art", "domaine_etude"),
        ("heure_supplementaires", "Non défini", "heure_supplementaires"),
        ("frequence_deplacement", "Rarement", "frequence_deplacement"),
        # Tests de pourcentage
        ("augementation_salaire_precedente", "5", "augementation_salaire_precedente"),
        ("augementation_salaire_precedente", "100%", "augementation_salaire_precedente"),
        ("augementation_salaire_precedente", "5.5%", "augementation_salaire_precedente"),
        ("augementation_salaire_precedente", "abc%", "augementation_salaire_precedente"),
    ],
)
def test_invalid_field_values_are_rejected(base_valid_payload, field, invalid_value, error_msg_part):
    """Les valeurs invalides pour chaque champ doivent lever une ValidationError."""
    payload = base_valid_payload.copy()
    payload[field] = invalid_value
    with pytest.raises(ValidationError) as exc_info:
        PredictionRequest(**payload)
    assert error_msg_part in str(exc_info.value)


@pytest.mark.parametrize(
    "field_to_remove, error_msg_part",
    [
        ("age", "age"),
        ("genre", "genre"),
        ("revenu_mensuel", "revenu_mensuel"),
        ("poste", "poste"),
        ("augementation_salaire_precedente", "augementation_salaire_precedente"),
    ],
)
def test_missing_required_fields_are_rejected(base_valid_payload, field_to_remove, error_msg_part):
    """Les champs requis manquants doivent lever une ValidationError."""
    payload = base_valid_payload.copy()
    del payload[field_to_remove]
    with pytest.raises(ValidationError) as exc_info:
        PredictionRequest(**payload)
    assert error_msg_part in str(exc_info.value)


@pytest.mark.parametrize(
    "annees_entreprise, annees_poste, annee_exp_totale, error_msg_part",
    [
        (5, 6, 10, "Années dans le poste"),   # > entreprise
        (10, 5, 9, "Expérience totale"),      # < entreprise
    ],
)
def test_coherence_validation_failures(base_valid_payload, annees_entreprise, annees_poste, annee_exp_totale, error_msg_part):
    """Les incohérences entre champs doivent lever une ValidationError."""
    payload = base_valid_payload.copy()
    payload["annees_dans_l_entreprise"] = annees_entreprise
    payload["annees_dans_le_poste_actuel"] = annees_poste
    payload["annee_experience_totale"] = annee_exp_totale
    with pytest.raises(ValidationError) as exc_info:
        PredictionRequest(**payload)
    assert error_msg_part in str(exc_info.value)