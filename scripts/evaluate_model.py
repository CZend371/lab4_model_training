import sys, os

sys.path.append(os.path.join(os.path.dirname(__file__), "../src"))

from ml_pipeline.data import load_data
from ml_pipeline.model import evaluate_model

if __name__ == "__main__":
    df = load_data("data/breast_cancer.csv")
    evaluate_model(df, "models/breast_cancer_model.pkl", "models/metrics.json")
