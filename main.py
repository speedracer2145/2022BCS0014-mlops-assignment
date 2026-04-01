from fastapi import FastAPI
import pickle
import numpy as np

app = FastAPI()

# Load model
model = pickle.load(open("outputs/model.pkl", "rb"))

# Health endpoint
@app.get("/")
def health():
    return {
        "name": "Alok P",
        "roll_no": "2022BCS0014"
    }

# Predict endpoint
@app.post("/predict")
def predict(data: dict):
    try:
        values = list(data.values())
        prediction = model.predict([values])[0]

        return {
            "prediction": float(prediction),
            "name": "Alok P",
            "roll_no": "2022BCS0014"
        }
    except Exception as e:
        return {"error": str(e)}
