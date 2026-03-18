from __future__ import annotations

from datetime import UTC, datetime
from typing import Any


def has_clinical_case(text: str) -> bool:
    if not text:
        return False

    lowered = text.lower()
    markers = [
        "cas clinique",
        "clinical case",
        "patient",
        "patiente",
    ]
    return any(marker in lowered for marker in markers)


def build_metadata(
    *,
    source: str,
    language: str,
    task_type: str,
    text_for_clinical_case: str = "",
    **extra: Any,
) -> dict:
    metadata = {
        "source": source,
        "language": language,
        "task_type": task_type,
        "has_clinical_case": has_clinical_case(text_for_clinical_case),
    }

    # Ajout dynamique des champs supplémentaires
    for key, value in extra.items():
        if value not in (None, "", [], {}):
            metadata[key] = value

    return metadata


def get_aggregated_at() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()