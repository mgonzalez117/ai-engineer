from __future__ import annotations

from typing import Any

from presidio_analyzer import AnalyzerEngine
from presidio_analyzer.nlp_engine import NlpEngineProvider
from presidio_anonymizer import AnonymizerEngine
from presidio_anonymizer.entities import OperatorConfig


TEXT_FIELDS_SFT = ("instruction", "input", "output")
TEXT_FIELDS_DPO = ("prompt", "chosen", "rejected")

# On limite volontairement les entités recherchées pour éviter
# les faux positifs sur le vocabulaire médical/biomédical.
ALLOWED_ENTITIES = [
    "PERSON",
    "EMAIL_ADDRESS",
    "PHONE_NUMBER",
    "IBAN_CODE",
    "CREDIT_CARD"
]

# Elimination du bruit
MIN_SCORE = 0.7

ANONYMIZER = AnonymizerEngine()

OPERATORS = {
    "PERSON": OperatorConfig("replace", {"new_value": "<PERSON>"}),
    "EMAIL_ADDRESS": OperatorConfig("replace", {"new_value": "<EMAIL>"}),
    "PHONE_NUMBER": OperatorConfig("replace", {"new_value": "<PHONE>"}),
    "IBAN_CODE": OperatorConfig("replace", {"new_value": "<IBAN>"}),
    "CREDIT_CARD": OperatorConfig("replace", {"new_value": "<CARD>"}),
}


def build_analyzer() -> AnalyzerEngine:
    configuration = {
        "nlp_engine_name": "spacy",
        "models": [
            {"lang_code": "fr", "model_name": "fr_core_news_md"},
            {"lang_code": "en", "model_name": "en_core_web_md"},
        ],
    }

    provider = NlpEngineProvider(nlp_configuration=configuration)
    nlp_engine = provider.create_engine()
    return AnalyzerEngine(nlp_engine=nlp_engine, supported_languages=["fr", "en"])


def analyze_text(analyzer: AnalyzerEngine, text: str, language: str):
    if not text:
        return []

    results = analyzer.analyze(
        text=text,
        language=language,
        entities=ALLOWED_ENTITIES,
    )
    return [r for r in results if r.score >= MIN_SCORE]


def anonymize_text(
    analyzer: AnalyzerEngine,
    text: str,
    language: str,
) -> tuple[str, list[dict[str, Any]]]:
    if not text:
        return "", []

    results = analyze_text(analyzer, text, language)

    if not results:
        return text, []

    anonymized = ANONYMIZER.anonymize(
        text=text,
        analyzer_results=results,
        operators=OPERATORS,
    )

    entities = [
        {
            "entity_type": r.entity_type,
            "start": r.start,
            "end": r.end,
            "score": r.score,
        }
        for r in results
    ]

    return anonymized.text, entities


def anonymize_record(
    record: dict[str, Any],
    text_fields: tuple[str, ...],
    analyzer: AnalyzerEngine,
) -> dict[str, Any]:
    language = record.get("language", "fr")
    if language not in {"fr", "en"}:
        language = "fr"

    output = dict(record)

    anonymization_details: dict[str, Any] = {
        "anonymized": False,
        "fields_modified": [],
        "detected_entities": {},
        "total_entities": 0,
    }

    for field in text_fields:
        original = output.get(field, "")
        if not isinstance(original, str):
            continue

        anonymized_text, entities = anonymize_text(analyzer, original, language)

        if entities:
            anonymization_details["detected_entities"][field] = entities
            anonymization_details["total_entities"] += len(entities)

        if anonymized_text != original:
            output[field] = anonymized_text
            anonymization_details["anonymized"] = True
            anonymization_details["fields_modified"].append(field)

    metadata = dict(output.get("metadata", {}))
    metadata["anonymization"] = anonymization_details
    output["metadata"] = metadata

    return output


def anonymize_sft_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    analyzer = build_analyzer()
    return [anonymize_record(row, TEXT_FIELDS_SFT, analyzer) for row in rows]


def anonymize_dpo_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    analyzer = build_analyzer()
    return [anonymize_record(row, TEXT_FIELDS_DPO, analyzer) for row in rows]