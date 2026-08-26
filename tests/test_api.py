from fastapi.testclient import TestClient
from app import app

client = TestClient(app)

def test_health_ok():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
    
def test_predict_returns_200(patch_load_pipeline):    
    response = client.post("/predict", json={
        "Store": 1,
        "Date": "2015-02-15",
        "Promo": 1,
        "StateHoliday": "0",
        "SchoolHoliday": 0
    })
    assert response.status_code == 200
    assert "predicted_sales" in response.json()
    assert response.json()["predicted_sales"] >= 0
    
def test_predict_unknown_store_returns_400(patch_load_pipeline):
    response = client.post("/predict", json={
        "Store": 9999, "Date": "2015-02-15", "Promo": 1,
        "StateHoliday": "0", "SchoolHoliday": 0,
    })
    assert response.status_code == 400


def test_predict_bad_type_returns_422():
    response = client.post("/predict", json={
        "Store": "hello", "Date": "2015-02-15", "Promo": 1,
        "StateHoliday": "0", "SchoolHoliday": 0,
    })
    assert response.status_code == 422