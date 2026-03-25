import os
import json
import joblib
import pandas as pd
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

def train_model(df: pd.DataFrame, model_path: str = "models/breast_cancer_model.pkl") -> float:
    """Train a logistic regression classifier and save it."""
    X = df.drop(columns=["target"])
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    clf = LogisticRegression(max_iter=200)
    clf.fit(X_train, y_train)

    preds = clf.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"[ml_pipeline.model] Model accuracy: {acc:.4f}")

    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(clf, model_path)
    print(f"[ml_pipeline.model] Saved model to {model_path}")

    return acc


def evaluate_model(
    df: pd.DataFrame,
    model_path: str = "models/breast_cancer_model.pkl",
    metrics_path: str = "models/metrics.json",
) -> dict:
    """Load a trained model, compute test accuracy, and save metrics to JSON."""
    X = df.drop(columns=["target"])
    y = df["target"]

    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    clf = joblib.load(model_path)
    predictions = clf.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    metrics = {"accuracy": round(accuracy, 4)}

    os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"[ml_pipeline.model] Metrics saved to {metrics_path}: {metrics}")
    return metrics


def save_metadata(
    accuracy: float,
    model_version: str | None = None,
    dataset: str = "breast_cancer",
    model_type: str = "logistic_regression",
    metadata_path: str = "models/metadata.json",
) -> dict:
    """Generate a version identifier and save model metadata to JSON."""
    if model_version is None:
        model_version = datetime.now().strftime("%Y%m%d_%H%M%S")

    metadata = {
        "model_version": model_version,
        "dataset": dataset,
        "model_type": model_type,
        "accuracy": round(accuracy, 4),
    }

    os.makedirs(os.path.dirname(metadata_path), exist_ok=True)
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"[ml_pipeline.model] Metadata saved to {metadata_path}: {metadata}")
    return metadata