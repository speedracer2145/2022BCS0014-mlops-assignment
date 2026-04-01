import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import pickle
import json
import os

# Load dataset
df = pd.read_csv("data/housing.csv")

# Assume last column is target
X = df.iloc[:, :-1]
y = df.iloc[:, -1]

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Start MLflow
mlflow.set_experiment("2022BCS0014_experiment")

with mlflow.start_run():

    model = LinearRegression()
    model.fit(X_train, y_train)

    preds = model.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)

    # Log params + metrics
    mlflow.log_param("model", "LinearRegression")
    mlflow.log_metric("rmse", rmse)
    mlflow.log_metric("r2", r2)

    # Save model
    os.makedirs("outputs", exist_ok=True)
    pickle.dump(model, open("outputs/model.pkl", "wb"))

    # Save metrics (assignment requirement)
    metrics = {
        "rmse": rmse,
        "r2": r2,
        "name": "Alok P",
        "roll_no": "2022BCS0014"
    }

    with open("outputs/metrics.json", "w") as f:
        json.dump(metrics, f)

    mlflow.sklearn.log_model(model, "model")

print("Training complete")
