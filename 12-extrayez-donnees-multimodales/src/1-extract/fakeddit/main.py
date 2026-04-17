import os
import json
import time
from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from src.models.models import EtlMetrics

from .download import download_fakeddit
from .extract import iter_tsv, extract_record


def raw_data_exists(path: Path) -> bool:
    return path.exists() and any(path.iterdir())


def main():
    start_time = time.time()

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

    duration = time.time() - start_time
    rows_per_sec = count_out / duration if duration > 0 else 0

    print(f"Read: {count_in}")
    print(f"Wrote: {count_out}")
    print(f"Rejected: {count_rejected}")
    print(f"→ rejected log: {log_path}")

    # Log metrics ONLY when executed by Airflow
    if os.getenv("AIRFLOW_CTX_DAG_ID"):
        engine = create_engine(os.environ["AIRFLOW__DATABASE__SQL_ALCHEMY_CONN"])
        Session = sessionmaker(bind=engine)

        with Session() as session:
            metric = EtlMetrics(
                pipeline_name=os.getenv("AIRFLOW_CTX_DAG_ID", "fakeddit_pipeline"),
                step="extract",
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
        print("[fakeddit] Not an Airflow run: skipping etl_metrics insert")


if __name__ == "__main__":
    main()
