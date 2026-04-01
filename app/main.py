from fastapi import FastAPI
import joblib
import numpy as np

app = FastAPI()

NAME = "Alok P"
ROLL_NO = "2022BCS0014"

# Load model
model = joblib.load("outputs/model.pkl")


@app.get("/")
def health():
    return {
        "name": NAME,
        "roll_no": ROLL_NO
    }


@app.post("/predict")
def predict(features: list):
    prediction = model.predict([features]).tolist()

    return {
        "prediction": prediction,
        "name": NAME,
        "roll_no": ROLL_NO
    }
