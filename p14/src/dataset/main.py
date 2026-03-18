from __future__ import annotations

import random

from .io import PROCESSED_DIR, ensure_dirs, write_json, write_jsonl
from .metadata import get_aggregated_at
from .normalize import (
    frenchmedmcqa_to_sft,
    mediqal_to_sft,
    medquad_to_sft,
    ultramedical_to_dpo,
)


def split_dataset(rows: list[dict], seed: int = 42) -> dict[str, list[dict]]:
    shuffled = rows[:]
    rng = random.Random(seed)
    rng.shuffle(shuffled)

    n = len(shuffled)
    train_end = int(0.8 * n)
    val_end = int(0.9 * n)

    return {
        "train": shuffled[:train_end],
        "val": shuffled[train_end:val_end],
        "test": shuffled[val_end:],
    }


def count_by_dataset(rows: list[dict]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        dataset = row.get("dataset", "unknown")
        counts[dataset] = counts.get(dataset, 0) + 1
    return counts


def count_by_language(rows: list[dict]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        language = row.get("language", "unknown")
        counts[language] = counts.get(language, 0) + 1
    return counts


def sample_sft_balanced(rows: list[dict], n: int = 5000, seed: int = 42) -> list[dict]:
    """
    Échantillonnage équilibré 50/50 FR/EN si possible.
    Si une langue n'a pas assez d'exemples, on complète avec l'autre.
    """
    if len(rows) <= n:
        return rows

    rng = random.Random(seed)

    fr_rows = [row for row in rows if row.get("language") == "fr"]
    en_rows = [row for row in rows if row.get("language") == "en"]
    other_rows = [row for row in rows if row.get("language") not in {"fr", "en"}]

    target_fr = n // 2
    target_en = n - target_fr

    sampled_fr = rng.sample(fr_rows, min(target_fr, len(fr_rows)))
    sampled_en = rng.sample(en_rows, min(target_en, len(en_rows)))

    sampled = sampled_fr + sampled_en
    missing = n - len(sampled)

    if missing > 0:
        sampled_ids = {row["id"] for row in sampled}
        pool = [
            row for row in (fr_rows + en_rows + other_rows)
            if row["id"] not in sampled_ids
        ]
        if pool:
            sampled.extend(rng.sample(pool, min(missing, len(pool))))

    rng.shuffle(sampled)
    return sampled


def build_clinical_eval(rows: list[dict], max_examples: int = 200, seed: int = 42) -> list[dict]:
    """
    Construit un jeu clinique séparé à partir des exemples marqués
    has_clinical_case dans metadata.
    """
    clinical_rows = [
        row for row in rows
        if row.get("metadata", {}).get("has_clinical_case") is True
    ]

    if len(clinical_rows) <= max_examples:
        return clinical_rows

    return sample_sft_balanced(clinical_rows, n=max_examples, seed=seed)


def remove_rows_by_id(rows: list[dict], rows_to_remove: list[dict]) -> list[dict]:
    ids_to_remove = {row["id"] for row in rows_to_remove}
    return [row for row in rows if row["id"] not in ids_to_remove]


def main() -> None:
    ensure_dirs()
    aggregated_at = get_aggregated_at()

    # 1) Agrégation complète
    all_sft_rows: list[dict] = []
    all_sft_rows.extend(medquad_to_sft())
    all_sft_rows.extend(mediqal_to_sft())
    all_sft_rows.extend(frenchmedmcqa_to_sft())

    all_dpo_rows = ultramedical_to_dpo()

    # 2) Sous-échantillonnage SFT à 5000 avec équilibre FR/EN
    sft_rows = sample_sft_balanced(all_sft_rows, n=5000, seed=42)

    # 3) Extraction du jeu clinique séparé AVANT les splits
    clinical_eval_rows = build_clinical_eval(sft_rows, max_examples=200, seed=42)

    # 4) Suppression du jeu clinique du pool principal
    remaining_sft_rows = remove_rows_by_id(sft_rows, clinical_eval_rows)

    # 5) Splits train / val / test sur le reste
    sft_splits = split_dataset(remaining_sft_rows, seed=42)
    dpo_splits = split_dataset(all_dpo_rows, seed=42)

    # 6) Écriture SFT
    write_jsonl(PROCESSED_DIR / "sft.jsonl", sft_rows)
    write_jsonl(PROCESSED_DIR / "train.jsonl", sft_splits["train"])
    write_jsonl(PROCESSED_DIR / "val.jsonl", sft_splits["val"])
    write_jsonl(PROCESSED_DIR / "test.jsonl", sft_splits["test"])
    write_jsonl(PROCESSED_DIR / "clinical_eval.jsonl", clinical_eval_rows)

    # 7) Écriture DPO
    write_jsonl(PROCESSED_DIR / "dpo.jsonl", all_dpo_rows)
    write_jsonl(PROCESSED_DIR / "dpo_train.jsonl", dpo_splits["train"])
    write_jsonl(PROCESSED_DIR / "dpo_val.jsonl", dpo_splits["val"])
    write_jsonl(PROCESSED_DIR / "dpo_test.jsonl", dpo_splits["test"])

    # 8) Manifest
    manifest = {
        "aggregated_at": aggregated_at,
        "sft_target_size": 5000,
        "clinical_eval_target_size": 200,
        "sources": [
            "medquad",
            "mediqal",
            "frenchmedmcqa",
            "ultramedical_preference",
        ],
        "files": [
            "sft.jsonl",
            "train.jsonl",
            "val.jsonl",
            "test.jsonl",
            "clinical_eval.jsonl",
            "dpo.jsonl",
            "dpo_train.jsonl",
            "dpo_val.jsonl",
            "dpo_test.jsonl",
            "stats.json",
            "manifest.json",
        ],
        "status": "ok",
    }
    write_json(PROCESSED_DIR / "manifest.json", manifest)

    # 9) Stats
    stats = {
        "aggregated_at": aggregated_at,
        "sft_total_before_sampling": len(all_sft_rows),
        "sft_total_after_sampling": len(sft_rows),
        "sft_languages_after_sampling": count_by_language(sft_rows),
        "sft_datasets_before_sampling": count_by_dataset(all_sft_rows),
        "sft_datasets_after_sampling": count_by_dataset(sft_rows),
        "clinical_eval_count": len(clinical_eval_rows),
        "clinical_eval_languages": count_by_language(clinical_eval_rows),
        "remaining_sft_after_clinical_split": len(remaining_sft_rows),
        "sft_splits": {
            "train": len(sft_splits["train"]),
            "val": len(sft_splits["val"]),
            "test": len(sft_splits["test"]),
        },
        "sft_split_languages": {
            "train": count_by_language(sft_splits["train"]),
            "val": count_by_language(sft_splits["val"]),
            "test": count_by_language(sft_splits["test"]),
        },
        "dpo_total": len(all_dpo_rows),
        "dpo_datasets": count_by_dataset(all_dpo_rows),
        "dpo_languages": count_by_language(all_dpo_rows),
        "dpo_splits": {
            "train": len(dpo_splits["train"]),
            "val": len(dpo_splits["val"]),
            "test": len(dpo_splits["test"]),
        },
        "status": "ok",
    }

    write_json(PROCESSED_DIR / "stats.json", stats)

    print(f"Aggregated at: {aggregated_at}")
    print(f"SFT total avant sampling: {len(all_sft_rows)}")
    print(f"SFT total après sampling: {len(sft_rows)}")
    print(f"SFT languages: {count_by_language(sft_rows)}")
    print(f"Clinical eval: {len(clinical_eval_rows)}")
    print(f"Clinical eval languages: {count_by_language(clinical_eval_rows)}")
    print(f"Remaining SFT after clinical split: {len(remaining_sft_rows)}")
    print(
        f"SFT splits -> train={len(sft_splits['train'])}, "
        f"val={len(sft_splits['val'])}, "
        f"test={len(sft_splits['test'])}"
    )
    print(f"DPO total: {len(all_dpo_rows)}")
    print(
        f"DPO splits -> train={len(dpo_splits['train'])}, "
        f"val={len(dpo_splits['val'])}, "
        f"test={len(dpo_splits['test'])}"
    )
    print(f"Fichiers écrits dans: {PROCESSED_DIR}")


if __name__ == "__main__":
    main()