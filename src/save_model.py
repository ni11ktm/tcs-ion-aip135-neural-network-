from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import joblib
import pandas as pd
import sklearn
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

from preprocess import FEATURE_NAMES, preprocess_dataframe
from train import DATA_PATH, build_pipeline

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = PROJECT_ROOT / "models"
MODEL_FILE = MODELS_DIR / "customer_buying_pipeline.pkl"
METADATA_FILE = MODELS_DIR / "model_metadata.json"


def main() -> None:
    frame = pd.read_csv(DATA_PATH)
    features = preprocess_dataframe(frame)
    target = frame["will_buy"].astype(int)

    x_train, x_test, y_train, y_test = train_test_split(
        features, target, test_size=0.2, random_state=42, stratify=target
    )
    eval_pipeline = build_pipeline()
    eval_pipeline.fit(x_train, y_train)
    y_pred = eval_pipeline.predict(x_test)
    _ = accuracy_score(y_test, y_pred)
    _ = f1_score(y_test, y_pred)

    final_pipeline = build_pipeline()
    final_pipeline.fit(features, target)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(final_pipeline, MODEL_FILE)

    accuracy_percent = round(float(accuracy_score(y_test, y_pred) * 100), 2)
    f1_percent = round(float(f1_score(y_test, y_pred, average="weighted") * 100), 2)

    metadata = {
        "model_type": "Perceptron",
        "sklearn_version": sklearn.__version__,
        "accuracy_on_test": accuracy_percent,
        "f1_score": f1_percent,
        "training_date": str(date.today()),
        "feature_names": FEATURE_NAMES,
        "target_classes": [0, 1],
        "class_labels": {"0": "Will Not Buy", "1": "Will Buy"},
        "model_version": "1.0.0",
    }
    METADATA_FILE.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"Saved model: {MODEL_FILE}")
    print(f"Saved metadata: {METADATA_FILE}")


if __name__ == "__main__":
    main()
