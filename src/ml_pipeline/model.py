import os
import json
import joblib
import pandas as pd
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
    predictionss = clf.predict(X_test)
    accuracy = accuracy_score(y_test, preds)

    metrics = {"accuracy": round(accuracy, 4)}

    os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"[ml_pipeline.model] Metrics saved to {metrics_path}: {metrics}")
    return metrics