import pytest
import src.predict as predict_mod
from src.predict import predict
from src.train import build_pipeline
from src.features import add_features
from tests.conftest import _raw_frame


@pytest.fixture
def patched(monkeypatch, store_reference):
    train = add_features(_raw_frame())
    pipe = build_pipeline().fit(train, train["Sales"])
    monkeypatch.setattr(predict_mod, "_load_pipeline", lambda: pipe)
    monkeypatch.setattr(predict_mod, "_load_store_reference", lambda: store_reference)


def _request():
    return [{"Store": 1, "Date": "2015-02-15", "Promo": 1,
             "StateHoliday": "0", "SchoolHoliday": 0}]


def test_predict_returns_euro_scale(patched):
    out = predict(_request())
    assert out.shape == (1,)
    assert 0 < out[0] < 50000


def test_missing_field_raises(patched):
    with pytest.raises(ValueError, match="missing required fields"):
        predict([{"Store": 1, "Date": "2015-02-15", "Promo": 1}])


def test_unknown_store_raises(patched):
    with pytest.raises(ValueError, match="unknown store"):
        predict([{"Store": 9999, "Date": "2015-02-15", "Promo": 1,
                  "StateHoliday": "0", "SchoolHoliday": 0}])