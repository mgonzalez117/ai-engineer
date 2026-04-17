from __future__ import annotations

from typing import Any

from .anonymize import TEXT_FIELDS_DPO, TEXT_FIELDS_SFT, build_analyzer, ALLOWED_ENTITIES

# Seuil plus bas que l'anonymisation
SCORE_THRESHOLD = 0.6

def serialize_analyzer_result(result: Any) -> dict[str, Any]:
    return {
        "entity_type": getattr(result, "entity_type", None),
        "start": getattr(result, "start", None),
        "end": getattr(result, "end", None),
        "score": getattr(result, "score", None),
    }


def validate_record(
    record: dict[str, Any],
    text_fields: tuple[str, ...],
    analyzer: Any,
    score_threshold: float = SCORE_THRESHOLD,
) -> dict[str, Any]:
    """
    Vérifie qu'il ne reste pas d'entités sensibles détectées
    dans les champs texte du record après anonymisation.
    """
    language = record.get("language", "fr")
    remaining_entities: dict[str, list[dict[str, Any]]] = {}
    checked_fields: list[str] = []
    total_remaining = 0

    for field in text_fields:
        text = record.get(field, "")
        if not isinstance(text, str) or not text.strip():
            continue

        checked_fields.append(field)

        results = analyzer.analyze(
            text=text,
            language=language,
            score_threshold=score_threshold,
            entities=ALLOWED_ENTITIES,
        )

        if results:
            serialized = [serialize_analyzer_result(r) for r in results]
            remaining_entities[field] = serialized
            total_remaining += len(serialized)

    return {
        "id": record.get("id"),
        "dataset": record.get("dataset"),
        "language": language,
        "valid": total_remaining == 0,
        "checked_fields": checked_fields,
        "remaining_entities": remaining_entities,
        "remaining_entities_count": total_remaining,
    }


def validate_rows(
    rows: list[dict[str, Any]],
    text_fields: tuple[str, ...],
    score_threshold: float = SCORE_THRESHOLD,
) -> list[dict[str, Any]]:
    analyzer = build_analyzer()
    return [
        validate_record(
            record=row,
            text_fields=text_fields,
            analyzer=analyzer,
            score_threshold=score_threshold,
        )
        for row in rows
    ]


def validate_sft_rows(
    rows: list[dict[str, Any]],
    score_threshold: float = SCORE_THRESHOLD,
) -> list[dict[str, Any]]:
    return validate_rows(
        rows=rows,
        text_fields=TEXT_FIELDS_SFT,
        score_threshold=score_threshold,
    )


def validate_dpo_rows(
    rows: list[dict[str, Any]],
    score_threshold: float = SCORE_THRESHOLD,
) -> list[dict[str, Any]]:
    return validate_rows(
        rows=rows,
        text_fields=TEXT_FIELDS_DPO,
        score_threshold=score_threshold,
    )


def summarize_validation(results: list[dict[str, Any]]) -> dict[str, Any]:
    total_records = len(results)
    valid_records = sum(1 for r in results if r.get("valid") is True)
    invalid_records = total_records - valid_records

    invalid_by_dataset: dict[str, int] = {}
    invalid_examples: list[dict[str, Any]] = []

    for result in results:
        if result.get("valid") is True:
            continue

        dataset = result.get("dataset", "unknown")
        invalid_by_dataset[dataset] = invalid_by_dataset.get(dataset, 0) + 1

        if len(invalid_examples) < 20:
            invalid_examples.append(
                {
                    "id": result.get("id"),
                    "dataset": dataset,
                    "language": result.get("language"),
                    "remaining_entities_count": result.get("remaining_entities_count", 0),
                    "remaining_entity_types": {
                        field: sorted(
                            {
                                entity.get("entity_type")
                                for entity in entities
                                if entity.get("entity_type")
                            }
                        )
                        for field, entities in result.get("remaining_entities", {}).items()
                    },
                }
            )

    success_rate = valid_records / total_records if total_records else 0.0

    return {
        "total_records": total_records,
        "valid_records": valid_records,
        "invalid_records": invalid_records,
        "success_rate": success_rate,
        "invalid_by_dataset": invalid_by_dataset,
        "invalid_examples": invalid_examples,
    }