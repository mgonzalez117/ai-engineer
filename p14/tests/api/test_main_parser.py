from __future__ import annotations

from src.api.main import parse_json_triage


def test_parse_json_triage_uses_next_valid_candidate_when_first_is_invalid() -> None:
    raw_output = """
    Pas d'entete ni point virgule au debut de la ligne
    JSON complet pour l'instant
    {
      "niveau_urgence": "tous",
      "orientation": "",
      "justification": "",
      "garde_fou_active": true
    }
    {
      "niveau_urgence": "immediat",
      "orientation": "evaluation medicale immediate au service des urgences",
      "justification": "douleur thoracique avec instabilite hemodynamique",
      "garde_fou_active": false
    }
    """

    triage, parse_status_raw = parse_json_triage(raw_output)

    assert parse_status_raw == "ok_json"
    assert triage.niveau_urgence == "immediat"
    assert triage.garde_fou_active is False


def test_parse_json_triage_can_recover_from_plaintext_output() -> None:
    raw_output = (
        "orientation: evaluation medicale rapide au service des urgences\n"
        "justification: patient stable mais symptomes necessitant un avis rapide\n"
        "niveau urgence: urgent"
    )

    triage, parse_status_raw = parse_json_triage(raw_output)

    assert parse_status_raw == "ok_plaintext"
    assert triage.niveau_urgence == "urgent"
    assert triage.garde_fou_active is False
