# Lab 4: Model Training and Serving with Airflow + FastAPI

In this lab you will build an **end-to-end ML pipeline** using Apache Airflow and serve the trained model with FastAPI.

The pipeline includes:
1. **Generate Data** – downloads the breast cancer dataset and saves it as a CSV.
2. **Train Model** – trains a Logistic Regression classifier.
3. **Evaluate Model** – computes test accuracy and saves `models/metrics.json`.
4. **Save Metadata** – generates a version identifier and saves `models/metadata.json`.
5. **Promote Model** – checks the accuracy threshold and uploads artifacts to S3.
6. **Serve Model** – starts a FastAPI app for inference.

---

## 📂 Project Structure

```
lab4_model_training/
├── dags/
│   ├── ml_training_pipeline_v2     # full pipeline: generate → train → evaluate → metadata → promote
│   ├── ml_pipeline_dag.py          # legacy pipeline: generate + train only
│   ├── generate_data_dag.py        # generate dataset only
│   ├── train_model_dag.py          # train model only
│   ├── evaluate_model_dag.py       # evaluate model only
│   └── promoto_model_dag.py        # promote model only
├── src/
│   ├── ml_pipeline/
│   │   ├── data.py                 # data generation and loading
│   │   └── model.py                # train, evaluate, metadata, promote functions
│   └── app/
│       └── api.py                  # FastAPI application
├── scripts/
│   ├── generate_data.py            # CLI: generate dataset
│   ├── train_model.py              # CLI: train model
│   ├── evaluate_model.py           # CLI: evaluate model + save metrics and metadata
│   ├── promote_model.py            # CLI: promote model to S3
│   └── serve_api.py                # CLI: start FastAPI server
├── data/
│   └── breast_cancer.csv           # generated dataset
├── models/
│   ├── breast_cancer_model.pkl     # trained model
│   ├── metrics.json                # evaluation results
│   └── metadata.json               # model version and metadata
├── airflow_home/                   # Airflow metadata (created after setup)
├── requirements.txt
└── setup_airflow.sh                # one-time setup script
```

---

## 🛠 Environment Setup

We use **one virtual environment** for all labs.

1. Create and activate:

```
python3 -m venv ~/venvs/airflow-class
source ~/venvs/airflow-class/bin/activate
```

2. Install dependencies:

```
pip install -r requirements.txt
```

> The `requirements.txt` pins **Airflow 2.10.2** with `constraints-3.12.txt`. If you are on a different Python version, update the constraints URL accordingly (e.g. `constraints-3.10.txt` or `constraints-3.11.txt`).

---

## ⚙️ Airflow Setup (one time)

Run the setup script to initialize the Airflow database and configure the project:

```bash
source ./setup_airflow.sh
```

Then create an admin user:

```bash
airflow users create \
  --username admin \
  --password admin \
  --firstname Admin \
  --lastname User \
  --role Admin \
  --email admin@example.com
```

After setup, export `AIRFLOW_HOME` in every new terminal session (or add it to your shell profile):

```bash
export AIRFLOW_HOME=$(pwd)/airflow_home
export AIRFLOW__CORE__DAGS_FOLDER=$(pwd)/dags
```

---

## 🚀 Running Airflow

Use two terminals, each with the virtual environment activated and `AIRFLOW_HOME` exported:

**Terminal 1 – Scheduler**
```
source ~/venvs/airflow-class/bin/activate
export AIRFLOW_HOME=$(pwd)/airflow_home
airflow scheduler
```

**Terminal 2 – Webserver**
```
source ~/venvs/airflow-class/bin/activate
export AIRFLOW_HOME=$(pwd)/airflow_home
airflow webserver --port 8080 --host 0.0.0.0
```

Then visit: http://\<your-ec2-ip\>:8080
Login: `admin / admin`

> Make sure port **8080** is open in your EC2 security group inbound rules.

---

## 📊 DAGs to Explore

| DAG ID | Description |
|---|---|
| `ml_training_pipeline_v2` | Full pipeline: generate → train → evaluate → metadata → promote |
| `ml_pipeline` | Legacy pipeline: generate + train only |
| `generate_data_only` | Generate `data/breast_cancer.csv` |
| `train_model_only` | Train and save `models/breast_cancer_model.pkl` |
| `evaluate_model_only` | Evaluate model and save `models/metrics.json` |
| `promote_model_only` | Check threshold and upload artifacts to S3 |

---

## Running Scripts Without Airflow

You can run each step individually from the command line:

```bash
# 1. Generate dataset
python scripts/generate_data.py

# 2. Train model
python scripts/train_model.py

# 3. Evaluate model (saves metrics.json and metadata.json)
python scripts/evaluate_model.py

# 4. Promote model to S3 (requires S3_BUCKET_NAME env var)
export S3_BUCKET_NAME=your-bucket-name
python scripts/promote_model.py
```

---

## Serving the Model with FastAPI

After training the model, start the API server:

```bash
python scripts/serve_api.py
```

Then visit: http://\<your-ec2-ip\>:8000/docs

> Make sure port **8000** is open in your EC2 security group inbound rules.

### Endpoints

#### `GET /model/info`

Returns the loaded model's metadata:

```json
{
  "model_version": "20260316_153000",
  "dataset": "breast_cancer",
  "model_type": "logistic_regression",
  "accuracy": 0.9561
}
```

#### `POST /predict`

Accepts all 30 breast cancer features as a flat list:

```json
{
  "features": [17.99, 10.38, 122.8, 1001.0, 0.1184, 0.2776, 0.3001, 0.1471,
               0.2419, 0.07871, 1.095, 0.9053, 8.589, 153.4, 0.006399, 0.04904,
               0.05373, 0.01587, 0.03003, 0.006193, 25.38, 17.33, 184.6, 2019.0,
               0.1622, 0.6656, 0.7119, 0.2654, 0.4601, 0.1189]
}
```

Response:

```json
{
  "prediction": "malignant",
  "class_index": 0
}
```

The model predicts one of two classes:

| class_index | prediction |
|---|---|
| 0 | malignant |
| 1 | benign |

---

## Model Promotion

The `promote_model` step enforces a quality gate: the model is only uploaded to S3 if its accuracy meets the threshold (`>= 0.94`). If the accuracy is below the threshold, the task raises an error and the pipeline fails.

Artifacts are uploaded to:

```
s3://<your-bucket>/models/<model_version>/
    model.pkl
    metrics.json
    metadata.json
```

---

## ✅ Summary

By the end of this lab you will have:

- Built a full ML pipeline with Airflow (data → train → evaluate → version → promote).
- Produced a trained model artifact with evaluation metrics and a version identifier.
- Enforced a quality threshold before publishing the model.
- Served the trained model with FastAPI, including a `/model/info` metadata endpoint.
- Sent live inference requests to the breast cancer classifier.
