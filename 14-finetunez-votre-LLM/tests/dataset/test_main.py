from __future__ import annotations

import src.dataset.main as main


def test_main_orchestrates_pipeline_without_real_datasets(monkeypatch):
    written_jsonl: dict[str, list[dict]] = {}
    written_json: dict[str, dict] = {}

    medquad_rows = [
        {
            "id": "medquad-1",
            "dataset": "medquad",
            "language": "en",
            "instruction": "instr",
            "input": "input",
            "output": "output",
            "metadata": {"has_clinical_case": False},
        }
    ]
    mediqal_rows = [
        {
            "id": "mediqal-1",
            "dataset": "mediqal",
            "language": "fr",
            "instruction": "instr",
            "input": "input",
            "output": "output",
            "metadata": {"has_clinical_case": True},
        }
    ]
    french_rows = [
        {
            "id": "french-1",
            "dataset": "frenchmedmcqa",
            "language": "fr",
            "instruction": "instr",
            "input": "input",
            "output": "output",
            "metadata": {"has_clinical_case": False},
        }
    ]
    ultra_rows = [
        {
            "id": "ultra-1",
            "dataset": "ultramedical_preference",
            "language": "en",
            "prompt": "prompt",
            "chosen": "chosen",
            "rejected": "rejected",
            "metadata": {"has_clinical_case": False},
        }
    ]

    monkeypatch.setattr(main, "medquad_to_sft", lambda: medquad_rows)
    monkeypatch.setattr(main, "mediqal_to_sft", lambda: mediqal_rows)
    monkeypatch.setattr(main, "frenchmedmcqa_to_sft", lambda: french_rows)
    monkeypatch.setattr(main, "ultramedical_to_dpo", lambda: ultra_rows)

    monkeypatch.setattr(main, "ensure_dirs", lambda: None)
    monkeypatch.setattr(main, "get_aggregated_at", lambda: "2026-03-19T10:00:00+00:00")
    monkeypatch.setattr(main, "sample_sft_balanced", lambda rows, n, seed: rows)

    anonymize_sft_called = {"value": False}
    anonymize_dpo_called = {"value": False}
    validate_sft_called = {"value": False}
    validate_dpo_called = {"value": False}

    def fake_anonymize_sft_rows(rows):
        anonymize_sft_called["value"] = True
        return rows

    def fake_anonymize_dpo_rows(rows):
        anonymize_dpo_called["value"] = True
        return rows

    def fake_validate_sft_rows(rows):
        validate_sft_called["value"] = True
        return [
            {"id": row["id"], "dataset": row["dataset"], "language": row["language"], "valid": True}
            for row in rows
        ]

    def fake_validate_dpo_rows(rows):
        validate_dpo_called["value"] = True
        return [
            {"id": row["id"], "dataset": row["dataset"], "language": row["language"], "valid": True}
            for row in rows
        ]

    monkeypatch.setattr(main, "anonymize_sft_rows", fake_anonymize_sft_rows)
    monkeypatch.setattr(main, "anonymize_dpo_rows", fake_anonymize_dpo_rows)
    monkeypatch.setattr(main, "validate_sft_rows", fake_validate_sft_rows)
    monkeypatch.setattr(main, "validate_dpo_rows", fake_validate_dpo_rows)

    monkeypatch.setattr(
        main,
        "summarize_validation",
        lambda results: {
            "total_records": len(results),
            "valid_records": len(results),
            "invalid_records": 0,
            "success_rate": 1.0,
            "invalid_by_dataset": {},
            "invalid_examples": [],
        },
    )

    def fake_write_jsonl(path, data):
        written_jsonl[path.name] = data

    def fake_write_json(path, data):
        written_json[path.name] = data

    monkeypatch.setattr(main, "write_jsonl", fake_write_jsonl)
    monkeypatch.setattr(main, "write_json", fake_write_json)

    main.main()

    assert anonymize_sft_called["value"] is True
    assert anonymize_dpo_called["value"] is True
    assert validate_sft_called["value"] is True
    assert validate_dpo_called["value"] is True

    assert set(written_jsonl.keys()) == {
        "sft.jsonl",
        "train.jsonl",
        "val.jsonl",
        "test.jsonl",
        "clinical_eval.jsonl",
        "dpo.jsonl",
        "dpo_train.jsonl",
        "dpo_val.jsonl",
        "dpo_test.jsonl",
        "sft_quarantine.jsonl",
        "dpo_quarantine.jsonl",
    }

    assert set(written_json.keys()) == {
        "sft_validation_report.json",
        "dpo_validation_report.json",
        "manifest.json",
        "stats.json",
    }

    assert len(written_jsonl["sft.jsonl"]) == 3
    assert len(written_jsonl["dpo.jsonl"]) == 1
    assert len(written_jsonl["clinical_eval.jsonl"]) == 1
    assert len(written_jsonl["sft_quarantine.jsonl"]) == 0
    assert len(written_jsonl["dpo_quarantine.jsonl"]) == 0

    stats = written_json["stats.json"]
    assert stats["sft_total_before_sampling"] == 3
    assert stats["sft_total_after_sampling_before_anonymization"] == 3
    assert stats["sft_total_after_validation"] == 3
    assert stats["sft_quarantine_count"] == 0
    assert stats["dpo_total_before_validation"] == 1
    assert stats["dpo_total_after_validation"] == 1
    assert stats["dpo_quarantine_count"] == 0

    manifest = written_json["manifest.json"]
    assert manifest["status"] == "ok"
    assert manifest["anonymization"]["enabled"] is True
    assert manifest["anonymization"]["validation_enabled"] is True