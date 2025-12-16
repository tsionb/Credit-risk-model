from fastapi import FastAPI, HTTPException
import joblib
import numpy as np
from src.api.pydantic_models import CreditRiskInput


app = FastAPI(title="Credit Risk Prediction API")

# Load trained model once
try:
    model = joblib.load("models/best_model.pkl")
except Exception as e:
    raise RuntimeError(f"Model loading failed: {e}")

@app.get("/")
def health_check():
    return {"status": "API is running"}

@app.post("/predict")
def predict(data: CreditRiskInput):
    try:
        X = np.array(data.features).reshape(1, -1)
        probability = model.predict_proba(X)[0][1]
        prediction = int(probability >= 0.5)

        return {
            "prediction": prediction,
            "risk_probability": round(probability, 4)
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
