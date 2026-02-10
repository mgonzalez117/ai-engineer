import os
import json
from pathlib import Path

from .download import download_fakeddit
from .extract import iter_tsv, extract_record


def raw_data_exists(path: Path) -> bool:
    return path.exists() and any(path.iterdir())


def main():

    raw_dir = Path(os.getenv("DATA_RAW_DIR", "data/raw"))
    processed_dir = Path(os.getenv("DATA_PROCESSED_DIR", "data/processed"))
    processed_dir.mkdir(parents=True, exist_ok=True)

    fakeddit_raw_dir = raw_dir / "fakeddit"

    if not raw_data_exists(fakeddit_raw_dir):
        print("[fakeddit] Raw data not found, downloading...")
        if not download_fakeddit(fakeddit_raw_dir):
            print("[fakeddit][ERROR] Cannot proceed without raw data")
            return

    in_path = fakeddit_raw_dir / "multimodal_only_samples/multimodal_train.tsv"
    out_path = processed_dir / "fakeddit_train.normalized.jsonl"
    log_path = processed_dir / "fakeddit_train.rejected.jsonl"

    if not in_path.exists():
        print(f"[fakeddit][ERROR] Input file not found: {in_path}")
        return

    count_in = 0
    count_out = 0
    count_rejected = 0

    with (
        out_path.open("w", encoding="utf-8") as out,
        log_path.open("w", encoding="utf-8") as log,
    ):
        for rec in iter_tsv(in_path):
            count_in += 1

            item, reason = extract_record(rec)
            if item is None:
                count_rejected += 1
                log.write(json.dumps({
                    "reason": reason,
                    "id": rec.get("id"),
                    "title": rec.get("title"),
                }, ensure_ascii=False) + "\n")
                continue

            out.write(json.dumps(item, ensure_ascii=False) + "\n")
            count_out += 1

    print(f"Read: {count_in}")
    print(f"Wrote: {count_out}")
    print(f"Rejected: {count_rejected}")
    print(f"→ rejected log: {log_path}")


if __name__ == "__main__":
    main()
