from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session", autouse=True)
def ensure_model_artifacts():
    model_file = PROJECT_ROOT / "models" / "customer_buying_pipeline.pkl"
    metadata_file = PROJECT_ROOT / "models" / "model_metadata.json"
    if not model_file.exists() or not metadata_file.exists():
        subprocess.run(
            ["python", "src/save_model.py"],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )


@pytest.fixture()
def client():
    from api.app import app

    app.config["TESTING"] = True
    with app.test_client() as test_client:
        yield test_client


def test_health_returns_200(client):
    response = client.get("/health")
    assert response.status_code == 200
    body = response.get_json()
    assert body["status"] == "ok"


def test_predict_valid_input_returns_prediction(client):
    payload = {
        "age": 34,
        "gender": "Female",
        "annual_income": 72000,
        "purchase_history": 14,
        "product_category": "Electronics",
        "loyalty_score": 7.8,
        "time_on_site": 18.5,
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    body = response.get_json()
    assert body["prediction"] in [0, 1]
    assert "label" in body


def test_predict_missing_field_returns_400(client):
    payload = {
        "age": 34,
        "gender": "Female",
        "annual_income": 72000,
        "purchase_history": 14,
        "product_category": "Electronics",
        "loyalty_score": 7.8,
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 400
