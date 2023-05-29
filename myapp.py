from fastapi import FastAPI
from pycaret.regression import load_model, predict_model
import pandas as pd

app = FastAPI()

model = load_model('final_model')

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.post("/predict")
def predict(data: dict):
    data = pd.DataFrame(data, index=[0])
    prediction = predict_model(model, data)
    return {"prediction": prediction.Label[0]}
