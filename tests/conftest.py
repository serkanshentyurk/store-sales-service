import numpy as np
import pandas as pd
import pytest


def _raw_frame(n_stores=6, days=40, seed=0):
    """A merged, cleaned train+store frame (open, sales>0), pre-add_features."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2015-01-01", periods=days, freq="D")
    rows = []
    for s in range(1, n_stores + 1):
        for d in dates:
            rows.append({
                "Store": s,
                "Date": d,
                "Sales": int(rng.integers(3000, 9000)),
                "Open": 1,
                "Promo": int(rng.integers(0, 2)),
                "StateHoliday": "0",
                "SchoolHoliday": int(rng.integers(0, 2)),
                "StoreType": ["a", "b", "c", "d"][s % 4],
                "Assortment": ["a", "b", "c"][s % 3],
                "CompetitionDistance": float(rng.integers(100, 20000)),
                "CompetitionOpenSinceMonth": np.nan if s == 1 else 9,
                "CompetitionOpenSinceYear": np.nan if s == 1 else 2010,
                "Promo2": s % 2,
            })
    return pd.DataFrame(rows)


@pytest.fixture
def raw_frame():
    return _raw_frame()


@pytest.fixture
def store_reference():
    df = _raw_frame().drop_duplicates("Store")
    return df[["Store", "StoreType", "Assortment", "CompetitionDistance",
               "CompetitionOpenSinceMonth", "CompetitionOpenSinceYear",
               "Promo2"]].reset_index(drop=True)