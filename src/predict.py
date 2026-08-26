"""Load the fitted pipeline and score raw prediction requests."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.features import add_features

MODEL_PATH = Path("models") / "pipeline.joblib"
STORE_PATH = Path("data") / "store.csv"

# Fields a caller actually knows at request time.
REQUEST_FIELDS = ["Store", "Date", "Promo", "StateHoliday", "SchoolHoliday"]


@lru_cache(maxsize=1)
def _load_pipeline():
    return joblib.load(MODEL_PATH)


@lru_cache(maxsize=1)
def _load_store_reference() -> pd.DataFrame:
    return pd.read_csv(STORE_PATH)


def predict(records: list[dict]) -> np.ndarray:
    """Score a list of raw requests; returns predicted Sales in euros."""
    req = pd.DataFrame(records)

    missing = set(REQUEST_FIELDS) - set(req.columns)
    if missing:
        raise ValueError(f"missing required fields: {sorted(missing)}")

    req["StateHoliday"] = req["StateHoliday"].astype(str)

    store_ref = _load_store_reference()
    unknown = set(req["Store"]) - set(store_ref["Store"])
    if unknown:
        raise ValueError(f"unknown store id(s): {sorted(unknown)}")

    df = req.merge(store_ref, on="Store", how="left")
    df = add_features(df)

    return _load_pipeline().predict(df)


if __name__ == "__main__":
    example = [{
        "Store": 1, "Date": "2015-08-05", "Promo": 1,
        "StateHoliday": "0", "SchoolHoliday": 1,
    }]
    print(predict(example))