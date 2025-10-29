import dill
import pandas as pd

from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional


# Загрузка модели
with open('sberauto_pipe.pkl', 'rb') as f:
    artifact = dill.load(f)

model = artifact['model']

app = FastAPI()


class SessionData(BaseModel):
    # Обязательные поля
    session_id: str
    client_id: str
    visit_date: str
    visit_time: str
    visit_number: int
    utm_medium: str
    device_category: str
    device_screen_resolution: str
    device_browser: str
    geo_country: str
    geo_city: str

    # Опциональные поля
    utm_source: Optional[str] = None
    utm_campaign: Optional[str] = None
    utm_adcontent: Optional[str] = None
    utm_keyword: Optional[str] = None
    device_os: Optional[str] = None
    device_brand: Optional[str] = None
    device_model: Optional[str] = None

class PredictionResponse(BaseModel):
    session_id: str
    prediction: int


@app.get('/status')
def status():
    return {"status": "OK"}


@app.get('/version')
def version():
    return artifact['metadata']


@app.post('/predict', response_model=PredictionResponse)
def predict(session: SessionData):
    df = pd.DataFrame([session.dict()])
    pred = model.predict(df)[0]
    return {
        "session_id": session.session_id,
        "prediction": int(pred)
    }