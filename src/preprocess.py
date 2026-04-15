from __future__ import annotations

from typing import Dict, Iterable, Tuple

import pandas as pd

PRODUCT_CATEGORIES: Tuple[str, ...] = (
    "Electronics",
    "Personal Care",
    "Clothing",
    "Home",
    "Food",
)

FEATURE_NAMES = [
    "age",
    "gender_encoded",
    "annual_income",
    "purchase_history",
    "loyalty_score",
    "time_on_site",
    "product_cat_electronics",
    "product_cat_personal_care",
    "product_cat_clothing",
    "product_cat_home",
    "product_cat_food",
]


def _encode_gender(value: str) -> int:
    gender = str(value).strip().lower()
    if gender == "female":
        return 1
    if gender == "male":
        return 0
    raise ValueError("gender must be either 'Male' or 'Female'")


def _category_flags(category: str) -> Dict[str, int]:
    normalized = str(category).strip().lower()
    flags = {}
    for item in PRODUCT_CATEGORIES:
        key = f"product_cat_{item.lower().replace(' ', '_')}"
        flags[key] = int(normalized == item.lower())
    return flags


def preprocess_dataframe(frame: pd.DataFrame) -> pd.DataFrame:
    required = [
        "age",
        "gender",
        "annual_income",
        "purchase_history",
        "product_category",
        "loyalty_score",
        "time_on_site",
    ]
    missing = [col for col in required if col not in frame.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    processed = pd.DataFrame(index=frame.index)
    processed["age"] = frame["age"].astype(float)
    processed["gender_encoded"] = frame["gender"].apply(_encode_gender).astype(float)
    processed["annual_income"] = frame["annual_income"].astype(float)
    processed["purchase_history"] = frame["purchase_history"].astype(float)
    processed["loyalty_score"] = frame["loyalty_score"].astype(float)
    processed["time_on_site"] = frame["time_on_site"].astype(float)

    category_frame = frame["product_category"].apply(_category_flags).apply(pd.Series).astype(float)
    processed = pd.concat([processed, category_frame], axis=1)
    return processed[FEATURE_NAMES]


def preprocess_single_input(payload: dict) -> pd.DataFrame:
    return preprocess_dataframe(pd.DataFrame([payload]))


def validate_payload(payload: dict, required_fields: Iterable[str]) -> None:
    for field in required_fields:
        if field not in payload:
            raise ValueError(f"Missing required field: {field}")
