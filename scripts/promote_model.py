import sys, os

sys.path.append(os.path.join(os.path.dirname(__file__), "../src"))

from ml_pipeline.model import promote_model

if __name__ == "__main__":
    s3_uri = promote_model()
    print(f"Promoted to {s3_uri}")
