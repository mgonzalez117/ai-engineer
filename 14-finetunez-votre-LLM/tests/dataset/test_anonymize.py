from __future__ import annotations

from types import SimpleNamespace

import src.dataset.anonymize as anonymize


class FakeAnalyzer:
    def __init__(self, mapping: dict[str, list[SimpleNamespace]] | None = None):
        self.mapping = mapping or {}
        self.calls: list[dict[str, object]] = []

    def analyze(self, text: str, language: str, entities=None):
        self.calls.append(
            {
                "text": text,
                "language": language,
                "entities": entities,
            }
        )
        return self.mapping.get(text, [])


def make_result(entity_type: str, start: int, end: int, score: float) -> SimpleNamespace:
    return SimpleNamespace(
        entity_type=entity_type,
        start=start,
        end=end,
        score=score,
    )


def test_analyze_text_returns_empty_list_for_empty_text():
    analyzer = FakeAnalyzer()

    assert anonymize.analyze_text(analyzer, "", "en") == []
    assert anonymize.analyze_text(analyzer, None, "en") == []


def test_analyze_text_filters_on_allowed_entities_and_min_score():
    text = "John email john@example.com"
    analyzer = FakeAnalyzer(
        {
            text: [
                make_result("PERSON", 0, 4, 0.95),
                make_result("EMAIL_ADDRESS", 11, 27, 0.80),
                make_result("PHONE_NUMBER", 5, 10, 0.40),  # sous le seuil
            ]
        }
    )

    results = anonymize.analyze_text(analyzer, text, "en")

    assert len(results) == 2
    assert [r.entity_type for r in results] == ["PERSON", "EMAIL_ADDRESS"]

    assert len(analyzer.calls) == 1
    assert analyzer.calls[0]["entities"] == anonymize.ALLOWED_ENTITIES
    assert analyzer.calls[0]["language"] == "en"


def test_anonymize_text_returns_original_text_when_no_entities():
    analyzer = FakeAnalyzer({"No pii here": []})

    text, entities = anonymize.anonymize_text(analyzer, "No pii here", "en")

    assert text == "No pii here"
    assert entities == []


def test_anonymize_text_replaces_detected_entities():
    text = "John can be reached at john@example.com"
    analyzer = FakeAnalyzer(
        {
            text: [
                make_result("PERSON", 0, 4, 0.99),
                make_result("EMAIL_ADDRESS", 23, 39, 0.99),
            ]
        }
    )

    anonymized_text, entities = anonymize.anonymize_text(analyzer, text, "en")

    assert "<PERSON>" in anonymized_text
    assert "<EMAIL>" in anonymized_text
    assert "John" not in anonymized_text
    assert "john@example.com" not in anonymized_text

    assert entities == [
        {"entity_type": "PERSON", "start": 0, "end": 4, "score": 0.99},
        {"entity_type": "EMAIL_ADDRESS", "start": 23, "end": 39, "score": 0.99},
    ]


def test_anonymize_record_updates_only_string_fields_and_metadata():
    record = {
        "id": "row-1",
        "language": "en",
        "instruction": "Hello",
        "input": 123,  # non-string: doit être ignoré
        "output": "Contact John at john@example.com",
        "metadata": {"source": "unit-test"},
    }

    analyzer = FakeAnalyzer(
        {
            "Hello": [],
            "Contact John at john@example.com": [
                make_result("PERSON", 8, 12, 0.95),
                make_result("EMAIL_ADDRESS", 16, 32, 0.96),
            ],
        }
    )

    result = anonymize.anonymize_record(record, anonymize.TEXT_FIELDS_SFT, analyzer)

    assert result["instruction"] == "Hello"
    assert result["input"] == 123
    assert "<PERSON>" in result["output"]
    assert "<EMAIL>" in result["output"]

    details = result["metadata"]["anonymization"]
    assert details["anonymized"] is True
    assert details["fields_modified"] == ["output"]
    assert details["total_entities"] == 2
    assert "output" in details["detected_entities"]

    # vérifie qu'on a conservé le metadata existant
    assert result["metadata"]["source"] == "unit-test"


def test_anonymize_record_defaults_to_french_for_unknown_language():
    record = {
        "id": "row-2",
        "language": "de",
        "instruction": "Bonjour",
        "input": "",
        "output": "",
    }
    analyzer = FakeAnalyzer({"Bonjour": []})

    anonymize.anonymize_record(record, anonymize.TEXT_FIELDS_SFT, analyzer)

    assert analyzer.calls[0]["language"] == "fr"


def test_anonymize_record_marks_not_anonymized_when_nothing_changes():
    record = {
        "id": "row-3",
        "language": "en",
        "instruction": "No pii",
        "input": "",
        "output": "",
    }
    analyzer = FakeAnalyzer({"No pii": []})

    result = anonymize.anonymize_record(record, anonymize.TEXT_FIELDS_SFT, analyzer)

    details = result["metadata"]["anonymization"]
    assert details["anonymized"] is False
    assert details["fields_modified"] == []
    assert details["detected_entities"] == {}
    assert details["total_entities"] == 0


def test_anonymize_sft_rows_uses_build_analyzer(monkeypatch):
    analyzer = FakeAnalyzer(
        {
            "John": [make_result("PERSON", 0, 4, 0.99)],
            "Nothing": [],
            "Done": [],
        }
    )

    monkeypatch.setattr(anonymize, "build_analyzer", lambda: analyzer)

    rows = [
        {
            "id": "sft-1",
            "language": "en",
            "instruction": "John",
            "input": "Nothing",
            "output": "Done",
        }
    ]

    results = anonymize.anonymize_sft_rows(rows)

    assert len(results) == 1
    assert results[0]["instruction"] == "<PERSON>"
    assert results[0]["metadata"]["anonymization"]["anonymized"] is True


def test_anonymize_dpo_rows_uses_build_analyzer(monkeypatch):
    analyzer = FakeAnalyzer(
        {
            "John": [make_result("PERSON", 0, 4, 0.99)],
            "john@example.com": [make_result("EMAIL_ADDRESS", 0, 16, 0.99)],
            "Nothing": [],
        }
    )

    monkeypatch.setattr(anonymize, "build_analyzer", lambda: analyzer)

    rows = [
        {
            "id": "dpo-1",
            "language": "en",
            "prompt": "John",
            "chosen": "john@example.com",
            "rejected": "Nothing",
        }
    ]

    results = anonymize.anonymize_dpo_rows(rows)

    assert len(results) == 1
    assert results[0]["prompt"] == "<PERSON>"
    assert results[0]["chosen"] == "<EMAIL>"
    assert results[0]["rejected"] == "Nothing"
    assert results[0]["metadata"]["anonymization"]["fields_modified"] == ["prompt", "chosen"]