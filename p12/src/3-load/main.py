import os
from pathlib import Path

from .load import load_publications_jsonl


def main():
    final_dir = Path(os.getenv("DATA_FINAL_DIR", "data/final"))

    input_path = final_dir / "fakeddit_train.final.jsonl"

    if not input_path.exists():
        raise FileNotFoundError(f"File not found: {input_path}")

    load_publications_jsonl(str(input_path))


if __name__ == "__main__":
    main()
