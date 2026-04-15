from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from preprocess import preprocess_dataframe

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "customer_data_sample.csv"


def build_pipeline() -> Pipeline:
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "model",
                Perceptron(
                    max_iter=1000,
                    eta0=0.01,
                    tol=1e-3,
                    class_weight="balanced",
                    random_state=42,
                ),
            ),
        ]
    )


def train_model():
    frame = pd.read_csv(DATA_PATH)
    x = preprocess_dataframe(frame)
    y = frame["will_buy"].astype(int)

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42, stratify=y
    )

    pipeline = build_pipeline()
    pipeline.fit(x_train, y_train)

    y_train_pred = pipeline.predict(x_train)
    y_pred = pipeline.predict(x_test)

    train_acc = accuracy_score(y_train, y_train_pred)
    test_acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print(f"Training Accuracy: {train_acc*100:.2f}%")
    print(f"Test Accuracy:     {test_acc*100:.2f}%")
    print(classification_report(y_test, y_pred))

    return pipeline, train_acc, test_acc, f1


if __name__ == "__main__":
    train_model()
