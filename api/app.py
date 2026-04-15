from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
from flask import Flask, jsonify, request
from flask_cors import CORS

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocess import preprocess_single_input, validate_payload

MODEL_PATH = PROJECT_ROOT / "models" / "customer_buying_pipeline.pkl"
METADATA_PATH = PROJECT_ROOT / "models" / "model_metadata.json"
REQUIRED_FIELDS = [
    "age",
    "gender",
    "annual_income",
    "purchase_history",
    "product_category",
    "loyalty_score",
    "time_on_site",
]

app = Flask(__name__)
CORS(app)

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")
if not METADATA_PATH.exists():
    raise FileNotFoundError(f"Metadata file not found: {METADATA_PATH}")

MODEL = joblib.load(MODEL_PATH)
MODEL_METADATA = json.loads(METADATA_PATH.read_text(encoding="utf-8"))


def preprocess_input(data: dict):
    return preprocess_single_input(data)


def _validate_types(payload: dict) -> None:
    numeric_fields = ["age", "annual_income", "purchase_history", "loyalty_score", "time_on_site"]
    for field in numeric_fields:
        if not isinstance(payload[field], (int, float)):
            raise ValueError(f"Field '{field}' must be numeric")
    if not isinstance(payload["gender"], str):
        raise ValueError("Field 'gender' must be a string")
    if not isinstance(payload["product_category"], str):
        raise ValueError("Field 'product_category' must be a string")


@app.get("/health")
def health():
    return jsonify({"status": "ok", "model": "Perceptron", "accuracy": 99.51})


@app.get("/model-info")
def model_info():
    return jsonify(MODEL_METADATA)


@app.post("/predict")
def predict():
    payload = request.get_json(silent=True)
    if payload is None:
        return jsonify({"error": "Invalid JSON body"}), 400

    try:
        validate_payload(payload, REQUIRED_FIELDS)
        _validate_types(payload)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400

    try:
        transformed = preprocess_input(payload)
        prediction = int(MODEL.predict(transformed)[0])
        if hasattr(MODEL, "decision_function"):
            score = float(MODEL.decision_function(transformed)[0])
            probability = float(1.0 / (1.0 + np.exp(-score)))
        else:
            probability = float(prediction)

        label_map = MODEL_METADATA.get(
            "class_labels", {"0": "Will Not Buy", "1": "Will Buy"}
        )
        label = label_map.get(str(prediction), str(prediction))
        return jsonify(
            {
                "prediction": prediction,
                "label": label,
                "probability": round(probability, 2),
                "model_version": MODEL_METADATA.get("model_version", "1.0.0"),
                "input_received": payload,
            }
        )
    except Exception as exc:
        return jsonify({"error": f"Prediction failed: {exc}"}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
