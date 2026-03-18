from __future__ import annotations

from datetime import UTC, datetime


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
) -> dict:
    return {
        "source": source,
        "language": language,
        "task_type": task_type,
        "has_clinical_case": has_clinical_case(text_for_clinical_case),
    }


def get_aggregated_at() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()