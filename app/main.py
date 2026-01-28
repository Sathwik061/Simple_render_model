from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI(title="Simple ML Model API")

# Load model
model = joblib.load("app/model.pkl")

# Input schema
class InputData(BaseModel):
    features: list

@app.get("/")
def home():
    return {"message": "ML Model is running!"}

@app.post("/predict")
def predict(data: InputData):
    X = np.array(data.features).reshape(1, -1)
    prediction = model.predict(X)
    return {"prediction": int(prediction[0])}
