from datetime import datetime

from airflow import DAG
from airflow.operators.bash import BashOperator

default_args = {
    "owner": "p12",
    "retries": 1,
}

with DAG(
    dag_id="fakeddit_pipeline",
    default_args=default_args,
    description="Pipeline extract + transform + load Fakeddit",
    start_date=datetime(2026, 2, 1),
    schedule=None,
    catchup=False,
    tags=["p12", "fakeddit"],
) as dag:

    extract = BashOperator(
        task_id="extract_fakeddit",
        bash_command="""
        set -e
        cd /app
        export PYTHONPATH=/app
        python -m src.1-extract.fakeddit.main
        """,
    )

    transform = BashOperator(
        task_id="transform_fakeddit",
        bash_command="""
        set -e
        cd /app
        export PYTHONPATH=/app
        python -m src.2-transform.fakeddit.pipeline_transform
        """,
    )

    load = BashOperator(
        task_id="load_publications",
        bash_command="""
        set -e
        cd /app
        export PYTHONPATH=/app
        python -m src.3-load.main
        """,
    )

    extract >> transform >> load
