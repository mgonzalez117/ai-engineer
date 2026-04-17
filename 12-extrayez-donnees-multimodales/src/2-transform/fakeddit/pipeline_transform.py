import os
import json
import time
from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from src.models.models import EtlMetrics


def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def main():
    start_time = time.time()

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
    count_rejected = 0  # ici pas de rejet réel, mais cohérent pour metrics

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

    duration = time.time() - start_time
    rows_per_sec = count_out / duration if duration > 0 else 0

    print(f"[transform] Read: {count_in}")
    print(f"[transform] Written: {count_out}")
    print(f"[transform] Output: {out_path}")

    # Log metrics ONLY if Airflow run
    if os.getenv("AIRFLOW_CTX_DAG_ID"):
        engine = create_engine(os.environ["AIRFLOW__DATABASE__SQL_ALCHEMY_CONN"])
        Session = sessionmaker(bind=engine)

        with Session() as session:
            metric = EtlMetrics(
                pipeline_name=os.getenv("AIRFLOW_CTX_DAG_ID", "fakeddit_pipeline"),
                step="transform",
                run_id=os.getenv("AIRFLOW_CTX_DAG_RUN_ID"),
                execution_date=os.getenv("AIRFLOW_CTX_EXECUTION_DATE"),
                nb_input=count_in,
                nb_output=count_out,
                nb_rejected=count_rejected,
                duration_seconds=duration,
                rows_per_second=rows_per_sec,
                success=True,
            )
            session.add(metric)
            session.commit()
    else:
        print("[transform] Not an Airflow run: skipping etl_metrics insert")


if __name__ == "__main__":
    main()
