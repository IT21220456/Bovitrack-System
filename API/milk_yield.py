from fastapi import FastAPI, HTTPException, Form
from pydantic import BaseModel, validator
import numpy as np
import joblib
from typing import List

app = FastAPI()

# Load the trained model
try:
    milk_yield_model = joblib.load("model/milk_yield_model.joblib")
except Exception as e:
    raise RuntimeError("Failed to load the model") from e

# The original Pydantic model is preserved in case it's needed for JSON endpoints
class MilkYieldInput(BaseModel):
    # Expect exactly 4 features: Length of Lactation, Days Dry (Days), Peak Yield (Kg), Days To Peak (Days)
    features: List[float]

    @validator("features")
    def check_features_length(cls, v):
        if len(v) != 4:
            raise ValueError("features must have exactly 4 items")
        return v

# Prediction endpoint now uses form data inputs
@app.post("/predict")
def predict(
    lactation_length: float = Form(...),
    days_dry: float = Form(...),
    peak_yield: float = Form(...),
    days_to_peak: float = Form(...)
):
    try:
        # Convert form inputs to numpy array for prediction
        input_array = np.array([[lactation_length, days_dry, peak_yield, days_to_peak]])
        predicted_yield = milk_yield_model.predict(input_array)[0]
        threshold = 2000  # Modify this threshold as needed

        predicted_class = 1 if predicted_yield > threshold else 0

        return {"predicted_yield": predicted_yield, "predicted_class": predicted_class}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))