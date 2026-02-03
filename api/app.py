from http.client import HTTPException
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

from src.predict import predict_from_dataframe

MODEL_PATH = Path(__file__).parent.parent / 'models' / 'xgboost.joblib'
SCALER_PATH = Path(__file__).parent.parent / 'models' / 'xgboost_scaler.joblib'
THRESHOLD_PATH = Path(__file__).parent.parent / 'models' / 'xgboost_threshold.json'
MAX_BATCH_SIZE = 1000
app = FastAPI()
model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)


class Transaction(BaseModel):
    Time: float
    V1: float
    V2: float
    V3: float
    V4: float
    V5: float
    V6: float
    V7: float
    V8: float
    V9: float
    V10: float
    V11: float
    V12: float
    V13: float
    V14: float
    V15: float
    V16: float
    V17: float
    V18: float
    V19: float
    V20: float
    V21: float
    V22: float
    V23: float
    V24: float
    V25: float
    V26: float
    V27: float
    V28: float
    Amount: float


class BatchRequest(BaseModel):
    transactions: list[Transaction]


class PredictionResult(BaseModel):
    index: int
    fraud_probability: float
    recommendation: str


@app.post("/predict")
def predict(transaction: Transaction):
    data = pd.DataFrame([transaction.model_dump()])

    _, probas = predict_from_dataframe(model, scaler, data)

    return PredictionResult(
        index=0,
        fraud_probability=probas[0],
        recommendation="review" if probas > 0.3 else "approve"
    )


@app.post("/predict/batch")
def predict_batch(transactions: list[Transaction]):
    if len(transactions) > MAX_BATCH_SIZE:
        raise HTTPException(400, f'Max batch size is {transactions}')

    data = pd.DataFrame([t.model_dump() for t in transactions])
    _, probas = predict_from_dataframe(model, scaler, data)

    return [
        PredictionResult(
            index=i,
            fraud_probability=float(p),
            recommendation="review" if p > 0.3 else "approve"
        )
        for i, p in enumerate(probas)
    ]


@app.get('/health')
def health():
    return {'status': ' healthy'}
