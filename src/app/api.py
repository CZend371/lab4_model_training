import json
import joblib
import numpy as np
from pathlib import Path
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel


class BreastCancerRequest(BaseModel):
    features: list[float]


def create_app(
    model_path: str = "models/breast_cancer_model.pkl",
    metadata_path: str = "models/metadata.json",
):
    """Creates a FastAPI app that serves predictions for the breast cancer model."""
    if not Path(model_path).exists():
        raise RuntimeError(
            f"Model file not found at '{model_path}'. "
            "Train the model first (run the DAG or scripts/train_model.py)."
        )

    model = joblib.load(model_path)

    metadata = {}
    if Path(metadata_path).exists():
        with open(metadata_path) as f:
            metadata = json.load(f)

    app = FastAPI(title="Breast Cancer Model API")

    target_names = {0: "malignant", 1: "benign"}

    @app.get("/")
    def root():
        return {
            "message": "Breast cancer model is ready for inference!",
            "classes": target_names,
        }

    @app.get("/model/info")
    def model_info():
        return metadata

    @app.post("/predict")
    def predict(request: BreastCancerRequest):
        X = np.array([request.features])
        try:
            idx = int(model.predict(X)[0])
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))
        return {"prediction": target_names[idx], "class_index": idx}

    return app
