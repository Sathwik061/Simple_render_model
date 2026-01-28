from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import joblib

app = FastAPI(
    title="Simple ML API",
    description="ML model deployed on Render",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

model = joblib.load("app/model.pkl")

class InputData(BaseModel):
    features: list

@app.get("/")
def home():
    return {"message": "ML Model is running!"}

@app.post("/predict")
def predict(data: InputData):
    arr = np.array(data.features).reshape(1, -1)
    pred = model.predict(arr)
    return {"prediction": int(pred[0])}
