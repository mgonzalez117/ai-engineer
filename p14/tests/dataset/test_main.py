import copy

import pytest
import src.dataset.main as main_module


def make_row(
    row_id: str,
    dataset: str = "medquad",
    language: str = "fr",
    has_clinical_case: bool = False,
) -> dict:
    return {
        "id": row_id,
        "dataset": dataset,
        "language": language,
        "instruction": f"instruction-{row_id}",
        "response": f"response-{row_id}",
        "metadata": {
            "has_clinical_case": has_clinical_case,
        },
    }


def test_split_dataset_respects_80_10_10_and_conserves_all_rows():
    rows = [make_row(f"id_{i}") for i in range(10)]

    splits = main_module.split_dataset(rows, seed=123)

    assert len(splits["train"]) == 8
    assert len(splits["val"]) == 1
    assert len(splits["test"]) == 1

    all_ids = {row["id"] for row in rows}
    split_ids = {row["id"] for part in splits.values() for row in part}

    assert split_ids == all_ids
    assert len(split_ids) == 10


def test_count_by_dataset():
    rows = [
        make_row("1", dataset="medquad"),
        make_row("2", dataset="medquad"),
        make_row("3", dataset="mediqal"),
        {"id": "4"},  # dataset absent -> unknown
    ]

    counts = main_module.count_by_dataset(rows)

    assert counts == {
        "medquad": 2,
        "mediqal": 1,
        "unknown": 1,
    }


def test_count_by_language():
    rows = [
        make_row("1", language="fr"),
        make_row("2", language="fr"),
        make_row("3", language="en"),
        {"id": "4"},  # language absente -> unknown
    ]

    counts = main_module.count_by_language(rows)

    assert counts == {
        "fr": 2,
        "en": 1,
        "unknown": 1,
    }


def test_sample_sft_balanced_returns_original_rows_if_len_lte_n():
    rows = [make_row("1"), make_row("2")]

    sampled = main_module.sample_sft_balanced(rows, n=10, seed=42)

    assert sampled == rows


def test_sample_sft_balanced_balances_fr_en_when_possible():
    rows = (
        [make_row(f"fr_{i}", language="fr") for i in range(10)]
        + [make_row(f"en_{i}", language="en") for i in range(10)]
    )

    sampled = main_module.sample_sft_balanced(rows, n=10, seed=42)

    assert len(sampled) == 10
    counts = main_module.count_by_language(sampled)
    assert counts["fr"] == 5
    assert counts["en"] == 5


def test_sample_sft_balanced_completes_with_other_language_if_needed():
    rows = (
        [make_row(f"fr_{i}", language="fr") for i in range(2)]
        + [make_row(f"en_{i}", language="en") for i in range(10)]
    )

    sampled = main_module.sample_sft_balanced(rows, n=6, seed=42)

    assert len(sampled) == 6
    counts = main_module.count_by_language(sampled)

    # On prend 2 FR max puis on complète avec EN
    assert counts["fr"] == 2
    assert counts["en"] == 4


def test_build_clinical_eval_filters_has_clinical_case():
    rows = [
        make_row("1", has_clinical_case=True, language="fr"),
        make_row("2", has_clinical_case=False, language="fr"),
        make_row("3", has_clinical_case=True, language="en"),
    ]

    clinical = main_module.build_clinical_eval(rows, max_examples=10, seed=42)

    assert {row["id"] for row in clinical} == {"1", "3"}


def test_build_clinical_eval_limits_number_of_examples():
    rows = [
        make_row(f"fr_{i}", has_clinical_case=True, language="fr") for i in range(10)
    ] + [
        make_row(f"en_{i}", has_clinical_case=True, language="en") for i in range(10)
    ]

    clinical = main_module.build_clinical_eval(rows, max_examples=6, seed=42)

    assert len(clinical) == 6
    counts = main_module.count_by_language(clinical)
    assert counts["fr"] == 3
    assert counts["en"] == 3


def test_remove_rows_by_id():
    rows = [make_row("1"), make_row("2"), make_row("3")]
    rows_to_remove = [make_row("2")]

    remaining = main_module.remove_rows_by_id(rows, rows_to_remove)

    assert [row["id"] for row in remaining] == ["1", "3"]


def test_main_orchestrates_pipeline_without_real_datasets(monkeypatch):
    written_jsonl = {}
    written_json = {}

    fake_sft_rows = [
        make_row("sft_1", dataset="medquad", language="fr", has_clinical_case=True),
        make_row("sft_2", dataset="medquad", language="fr", has_clinical_case=False),
        make_row("sft_3", dataset="mediqal", language="en", has_clinical_case=True),
        make_row("sft_4", dataset="mediqal", language="en", has_clinical_case=False),
        make_row("sft_5", dataset="frenchmedmcqa", language="fr", has_clinical_case=False),
        make_row("sft_6", dataset="frenchmedmcqa", language="en", has_clinical_case=False),
    ]

    fake_dpo_rows = [
        {
            "id": "dpo_1",
            "dataset": "ultramedical_preference",
            "language": "en",
            "prompt": "prompt-1",
            "chosen": "good answer",
            "rejected": "bad answer",
        },
        {
            "id": "dpo_2",
            "dataset": "ultramedical_preference",
            "language": "fr",
            "prompt": "prompt-2",
            "chosen": "bonne réponse",
            "rejected": "mauvaise réponse",
        },
    ]

    monkeypatch.setattr(main_module, "ensure_dirs", lambda: None)
    monkeypatch.setattr(main_module, "get_aggregated_at", lambda: "2026-03-18T10:00:00+00:00")

    monkeypatch.setattr(main_module, "medquad_to_sft", lambda: copy.deepcopy(fake_sft_rows[:2]))
    monkeypatch.setattr(main_module, "mediqal_to_sft", lambda: copy.deepcopy(fake_sft_rows[2:4]))
    monkeypatch.setattr(main_module, "frenchmedmcqa_to_sft", lambda: copy.deepcopy(fake_sft_rows[4:6]))
    monkeypatch.setattr(main_module, "ultramedical_to_dpo", lambda: copy.deepcopy(fake_dpo_rows))

    def fake_write_jsonl(path, rows):
        written_jsonl[path.name] = copy.deepcopy(rows)

    def fake_write_json(path, payload):
        written_json[path.name] = copy.deepcopy(payload)

    monkeypatch.setattr(main_module, "write_jsonl", fake_write_jsonl)
    monkeypatch.setattr(main_module, "write_json", fake_write_json)

    main_module.main()

    expected_jsonl_files = {
        "sft.jsonl",
        "train.jsonl",
        "val.jsonl",
        "test.jsonl",
        "clinical_eval.jsonl",
        "dpo.jsonl",
        "dpo_train.jsonl",
        "dpo_val.jsonl",
        "dpo_test.jsonl",
    }
    assert set(written_jsonl.keys()) == expected_jsonl_files

    expected_json_files = {"manifest.json", "stats.json"}
    assert set(written_json.keys()) == expected_json_files

    # SFT complet
    assert len(written_jsonl["sft.jsonl"]) == 6

    # Jeu clinique extrait depuis SFT
    clinical_ids = {row["id"] for row in written_jsonl["clinical_eval.jsonl"]}
    assert clinical_ids == {"sft_1", "sft_3"}

    # Le reste est splitté
    remaining_count = 6 - 2
    assert len(written_jsonl["train.jsonl"]) == int(0.8 * remaining_count)
    assert len(written_jsonl["val.jsonl"]) == int(0.9 * remaining_count) - int(0.8 * remaining_count)
    assert len(written_jsonl["test.jsonl"]) == remaining_count - int(0.9 * remaining_count)

    split_ids = (
        {row["id"] for row in written_jsonl["train.jsonl"]}
        | {row["id"] for row in written_jsonl["val.jsonl"]}
        | {row["id"] for row in written_jsonl["test.jsonl"]}
    )
    assert split_ids == {"sft_2", "sft_4", "sft_5", "sft_6"}
    assert clinical_ids.isdisjoint(split_ids)

    # DPO split
    assert len(written_jsonl["dpo.jsonl"]) == 2
    assert len(written_jsonl["dpo_train.jsonl"]) == 1
    assert len(written_jsonl["dpo_val.jsonl"]) == 0
    assert len(written_jsonl["dpo_test.jsonl"]) == 1

    manifest = written_json["manifest.json"]
    assert manifest["aggregated_at"] == "2026-03-18T10:00:00+00:00"
    assert manifest["status"] == "ok"
    assert "stats.json" in manifest["files"]

    stats = written_json["stats.json"]
    assert stats["aggregated_at"] == "2026-03-18T10:00:00+00:00"
    assert stats["sft_total_before_sampling"] == 6
    assert stats["sft_total_after_sampling"] == 6
    assert stats["clinical_eval_count"] == 2
    assert stats["remaining_sft_after_clinical_split"] == 4
    assert stats["dpo_total"] == 2
    assert stats["status"] == "ok"