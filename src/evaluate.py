from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

from preprocess import preprocess_dataframe
from train import DATA_PATH, build_pipeline

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORTS_DIR = PROJECT_ROOT / "reports"


def evaluate_model() -> None:
    frame = pd.read_csv(DATA_PATH)
    x = preprocess_dataframe(frame)
    y = frame["will_buy"].astype(int)

    model = build_pipeline()
    model.fit(x, y)
    y_pred = model.predict(x)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    matrix = confusion_matrix(y, y_pred, labels=[0, 1])
    display = ConfusionMatrixDisplay(confusion_matrix=matrix, display_labels=[0, 1])
    display.plot(cmap="Blues")
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "confusion_matrix.png", dpi=150)
    plt.close()

    sns.histplot(frame["annual_income"], kde=True)
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "income_distribution.png", dpi=150)
    plt.close()


if __name__ == "__main__":
    evaluate_model()
