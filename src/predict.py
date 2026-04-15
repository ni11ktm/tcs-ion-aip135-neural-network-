from __future__ import annotations

from pathlib import Path

import joblib

from preprocess import preprocess_single_input

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "customer_buying_pipeline.pkl"


def predict_single(payload: dict) -> int:
    model = joblib.load(MODEL_PATH)
    transformed = preprocess_single_input(payload)
    return int(model.predict(transformed)[0])


if __name__ == "__main__":
    sample_payload = {
        "age": 34,
        "gender": "Female",
        "annual_income": 72000,
        "purchase_history": 14,
        "product_category": "Electronics",
        "loyalty_score": 7.8,
        "time_on_site": 18.5,
    }
    prediction = predict_single(sample_payload)
    print({"prediction": prediction})
