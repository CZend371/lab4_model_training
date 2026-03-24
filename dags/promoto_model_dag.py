from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
import sys, os

sys.path.append(os.path.join(os.path.dirname(__file__), "../src"))
from ml_pipeline.model import promote_model

default_args = {"owner": "airflow", "retries": 1}

with DAG(
    dag_id="promote_model_only",
    default_args=default_args,
    description="Promote ML model only (expects model to already exist)",
    schedule_interval=None,
    start_date=datetime(2025, 1, 1),
    catchup=False,
) as dag:

    def promote_model_wrapper(model_path: str):
        return promote_model(model_path)

    promote_task = PythonOperator(
        task_id="promote_model",
        python_callable=promote_model_wrapper,
        op_kwargs={
            "model_path": "models/breast_cancer_model.pkl",
        },
    )