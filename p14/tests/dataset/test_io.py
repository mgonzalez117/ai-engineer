from pathlib import Path

import pandas as pd

from src.dataset.io import read_table, write_json, write_jsonl


def test_write_jsonl_and_read_jsonl(tmp_path: Path) -> None:
    file_path = tmp_path / "sample.jsonl"

    rows = [
        {"id": 1, "text": "hello"},
        {"id": 2, "text": "world"},
    ]

    write_jsonl(file_path, rows)

    assert file_path.exists()

    df = read_table(file_path)

    assert isinstance(df, pd.DataFrame)
    assert len(df) == 2
    assert df.to_dict(orient="records") == rows


def test_write_json_creates_file(tmp_path: Path) -> None:
    file_path = tmp_path / "sample.json"

    data = {
        "status": "ok",
        "count": 2,
    }

    write_json(file_path, data)

    assert file_path.exists()
    assert file_path.read_text(encoding="utf-8").strip() != ""


def test_read_table_json(tmp_path: Path) -> None:
    file_path = tmp_path / "sample.json"
    file_path.write_text(
        '[{"id": 1, "value": "a"}, {"id": 2, "value": "b"}]',
        encoding="utf-8",
    )

    df = read_table(file_path)

    assert len(df) == 2
    assert list(df["id"]) == [1, 2]
    assert list(df["value"]) == ["a", "b"]


def test_read_table_csv(tmp_path: Path) -> None:
    file_path = tmp_path / "sample.csv"
    file_path.write_text(
        "id,value\n1,a\n2,b\n",
        encoding="utf-8",
    )

    df = read_table(file_path)

    assert len(df) == 2
    assert list(df["id"]) == [1, 2]
    assert list(df["value"]) == ["a", "b"]


def test_read_table_unsupported_format(tmp_path: Path) -> None:
    file_path = tmp_path / "sample.txt"
    file_path.write_text("hello", encoding="utf-8")

    try:
        read_table(file_path)
        assert False, "read_table should raise ValueError for unsupported formats"
    except ValueError as exc:
        assert "Unsupported file format" in str(exc)