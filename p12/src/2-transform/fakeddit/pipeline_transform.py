import os
import json
from pathlib import Path


def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def main():
    processed_dir = Path(os.getenv("DATA_PROCESSED_DIR", "data/processed"))
    final_dir = Path(os.getenv("DATA_FINAL_DIR", "data/final"))
    final_dir.mkdir(parents=True, exist_ok=True)

    in_path = processed_dir / "fakeddit_train.normalized.jsonl"
    out_path = final_dir / "fakeddit_train.final.jsonl"

    if not in_path.exists():
        print(f"[transform][ERROR] Input not found: {in_path}")
        return

    count_in = 0
    count_out = 0

    with out_path.open("w", encoding="utf-8") as out:
        for item in iter_jsonl(in_path):
            count_in += 1

            # nettoyage minimal
            if item.get("image_url"):
                item["image_url"] = item["image_url"].replace("&amp;", "&")

            # colonnes dérivées
            title = item.get("title", "")
            item["text_length"] = len(title)
            item["has_image"] = True

            out.write(json.dumps(item, ensure_ascii=False) + "\n")
            count_out += 1

    print(f"[transform] Read: {count_in}")
    print(f"[transform] Written: {count_out}")
    print(f"[transform] Output: {out_path}")


if __name__ == "__main__":
    main()
