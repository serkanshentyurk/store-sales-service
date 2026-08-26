import numpy as np
from src.features import add_features, build_preprocessor


def test_calendar_features_created(raw_frame):
    out = add_features(raw_frame)
    for col in ["Year", "Month", "Day", "WeekOfYear", "DayOfWeek", "CompetitionMonthsSince"]:
        assert col in out.columns


def test_dayofweek_matches_date(raw_frame):
    out = add_features(raw_frame)
    assert (out["DayOfWeek"] == out["Date"].dt.dayofweek + 1).all()


def test_competition_months_nan_preserved_for_missing(raw_frame):
    out = add_features(raw_frame)
    assert out.loc[out["Store"] == 1, "CompetitionMonthsSince"].isna().all()
    assert out.loc[out["Store"] != 1, "CompetitionMonthsSince"].notna().all()


def test_preprocessor_contract(raw_frame):
    out = add_features(raw_frame)
    Xt = build_preprocessor().fit_transform(out, out["Sales"])
    assert Xt.shape[0] == len(out)
    assert Xt.shape[1] > 0
    assert np.isfinite(Xt).all()