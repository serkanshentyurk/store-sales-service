from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from src.predict import predict

import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)

app = FastAPI(title = 'Store Sales Service')

class PredictionRequest(BaseModel):
    Store: int
    Date: str
    Promo: int
    StateHoliday: str
    SchoolHoliday: int
    
@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict_endpoint(request: PredictionRequest):
    try:
        prediction = predict([request.model_dump()])
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"predicted_sales": round(float(prediction[0]), 2)}