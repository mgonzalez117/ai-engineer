from src.dataset.metadata import build_metadata, has_clinical_case


def test_has_clinical_case_with_french_marker() -> None:
    text = "Cas clinique : patient de 65 ans avec douleur thoracique."
    assert has_clinical_case(text) is True


def test_has_clinical_case_with_english_marker() -> None:
    text = "Clinical case: patient with fever and cough."
    assert has_clinical_case(text) is True


def test_has_clinical_case_without_marker() -> None:
    text = "What is hypertension?"
    assert has_clinical_case(text) is False


def test_build_metadata() -> None:
    metadata = build_metadata(
        source="mediqal",
        language="fr",
        task_type="mcq",
        text_for_clinical_case="Cas clinique : patient présentant une fièvre.",
    )

    assert metadata["source"] == "mediqal"
    assert metadata["language"] == "fr"
    assert metadata["task_type"] == "mcq"
    assert metadata["has_clinical_case"] is True