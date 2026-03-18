from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


RAW_DIR = Path(".data/dataset/raw")
PROCESSED_DIR = Path(".data/dataset/processed")


def ensure_dirs() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def find_files(root: Path, extensions: tuple[str, ...] = (".json", ".jsonl", ".csv", ".parquet")) -> list[Path]:
    return sorted(
        p for p in root.rglob("*")
        if p.is_file() and p.suffix.lower() in extensions
    )


def read_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()

    if suffix == ".csv":
        return pd.read_csv(path)

    if suffix == ".jsonl":
        return pd.read_json(path, lines=True)

    if suffix == ".json":
        try:
            return pd.read_json(path)
        except ValueError:
            return pd.read_json(path, lines=True)

    if suffix == ".parquet":
        return pd.read_parquet(path)

    raise ValueError(f"Unsupported file format: {path}")


def find_first_table_with_columns(dataset_dir: Path, required_columns: set[str]) -> pd.DataFrame | None:
    for file_path in find_files(dataset_dir):
        try:
            df = read_table(file_path)
        except Exception:
            continue

        cols = {str(c) for c in df.columns}
        if required_columns.issubset(cols):
            return df

    return None


def find_all_tables_with_columns(dataset_dir: Path, required_columns: set[str]) -> list[pd.DataFrame]:
    matches: list[pd.DataFrame] = []

    for file_path in find_files(dataset_dir):
        try:
            df = read_table(file_path)
        except Exception:
            continue

        cols = {str(c) for c in df.columns}
        if required_columns.issubset(cols):
            matches.append(df)

    return matches


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)