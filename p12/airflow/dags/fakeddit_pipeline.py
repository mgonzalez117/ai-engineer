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
    description="Pipeline extract + transform Fakeddit",
    start_date=datetime(2026, 2, 1),
    schedule_interval=None,
    catchup=False,
    tags=["p12", "fakeddit"],
) as dag:

    extract = BashOperator(
        task_id="extract_fakeddit",
        bash_command="python -m src.1-extract.fakeddit.main",
    )

    transform = BashOperator(
        task_id="transform_fakeddit",
        bash_command="python -m src.2-transform.fakeddit.pipeline_transform",
    )

    extract >> transform
