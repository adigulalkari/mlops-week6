from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib

class IrisData(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

app = FastAPI()
model = joblib.load("model.joblib")

@app.get("/")
def home():
    return {"message": "Iris Classifier is live!"}

@app.post("/predict")
def predict(data: IrisData):
    input_df = pd.DataFrame([data.dict()])
    prediction = model.predict(input_df)[0]
    # return prediction
    return {"prediction": int(prediction)}


