import sys, os

# Ensure src is on the path
sys.path.append(os.path.join(os.path.dirname(__file__), "../src"))

from ml_pipeline.data import load_data
from ml_pipeline.model import train_model, save_metadata

if __name__ == "__main__":
    df = load_data("data/breast_cancer.csv")
    accuracy = train_model(df, "models/breast_cancer_model.pkl")
    save_metadata(accuracy=accuracy)