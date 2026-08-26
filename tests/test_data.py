import pandas as pd
from src.data import load_clean_data


def _write_csvs(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    pd.DataFrame({
        "Store": [1, 1, 1, 2], "DayOfWeek": [4, 5, 6, 4],
        "Date": ["2015-01-01", "2015-01-02", "2015-01-03", "2015-01-01"],
        "Sales": [5000, 0, 6000, 4000],      # open-but-zero -> drop
        "Customers": [500, 0, 600, 400],
        "Open": [1, 1, 0, 1],                # closed -> drop
        "Promo": [1, 1, 0, 0],
        "StateHoliday": ["0", "0", "0", "0"], "SchoolHoliday": [0, 0, 0, 1],
    }).to_csv(data / "train.csv", index=False)
    pd.DataFrame({
        "Store": [1, 2], "StoreType": ["c", "a"], "Assortment": ["a", "a"],
        "CompetitionDistance": [1270.0, 570.0],
        "CompetitionOpenSinceMonth": [9, 11], "CompetitionOpenSinceYear": [2008, 2007],
        "Promo2": [0, 1], "Promo2SinceWeek": [None, 13],
        "Promo2SinceYear": [None, 2010], "PromoInterval": ["", "Jan,Apr,Jul,Oct"],
    }).to_csv(data / "store.csv", index=False)
    return data


def test_drops_closed_and_zero_sales_rows(tmp_path):
    df = load_clean_data(str(_write_csvs(tmp_path)))
    assert len(df) == 2
    assert (df["Open"] == 1).all()
    assert (df["Sales"] > 0).all()


def test_customers_dropped_and_store_metadata_joined(tmp_path):
    df = load_clean_data(str(_write_csvs(tmp_path)))
    assert "Customers" not in df.columns
    assert "StoreType" in df.columns
    assert df["StoreType"].notna().all()


def test_state_holiday_not_parsed_as_numeric(tmp_path):
    df = load_clean_data(str(_write_csvs(tmp_path)))
    assert df["StateHoliday"].map(type).eq(str).all()
    assert not pd.api.types.is_numeric_dtype(df["StateHoliday"])