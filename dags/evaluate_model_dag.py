from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
import sys, os

sys.path.append(os.path.join(os.path.dirname(__file__), "../src"))

from ml_pipeline.data import load_data
from ml_pipeline.model import evaluate_model

default_args = {"owner": "airflow", "retries": 1}

with DAG(
    dag_id="evaluate_model_only",
    default_args=default_args,
    description="Evaluate ML model only (expects model to already exist)",
    schedule_interval=None,
    start_date=datetime(2025, 1, 1),
    catchup=False,
) as dag:

    def evaluate_model_wrapper(data_path: str, model_path: str, metrics_path: str):
        df = load_data(data_path)
        return evaluate_model(df, model_path, metrics_path)

    evaluate_task = PythonOperator(
        task_id="evaluate_model",
        python_callable=evaluate_model_wrapper,
        op_kwargs={
            "data_path": "data/breast_cancer.csv",
            "model_path": "models/breast_cancer_model.pkl",
            "metrics_path": "models/metrics.json",
        },
    )