import os
import time
from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from src.models.models import EtlMetrics
from .load import load_publications_jsonl


def main():
    start_time = time.time()

    final_dir = Path(os.getenv("DATA_FINAL_DIR", "data/final"))
    input_path = final_dir / "fakeddit_train.final.jsonl"

    if not input_path.exists():
        raise FileNotFoundError(f"File not found: {input_path}")

    nb_input, nb_output, nb_rejected = load_publications_jsonl(str(input_path))

    duration = time.time() - start_time
    rows_per_sec = nb_output / duration if duration > 0 else 0

    print(f"[load] Input: {nb_input}")
    print(f"[load] Output: {nb_output}")
    print(f"[load] Rejected: {nb_rejected}")
    print(f"[load] Duration: {duration:.2f}s")
    print(f"[load] Rows/sec: {rows_per_sec:.2f}")

    # Log metrics ONLY if Airflow run
    if os.getenv("AIRFLOW_CTX_DAG_ID"):
        engine = create_engine(os.environ["AIRFLOW__DATABASE__SQL_ALCHEMY_CONN"])
        Session = sessionmaker(bind=engine)

        with Session() as session:
            metric = EtlMetrics(
                pipeline_name=os.getenv("AIRFLOW_CTX_DAG_ID", "fakeddit_pipeline"),
                step="load",
                run_id=os.getenv("AIRFLOW_CTX_DAG_RUN_ID"),
                execution_date=os.getenv("AIRFLOW_CTX_EXECUTION_DATE"),
                nb_input=nb_input,
                nb_output=nb_output,
                nb_rejected=nb_rejected,
                duration_seconds=duration,
                rows_per_second=rows_per_sec,
                success=True,
            )
            session.add(metric)
            session.commit()
    else:
        print("[load] Not an Airflow run: skipping etl_metrics insert")


if __name__ == "__main__":
    main()
