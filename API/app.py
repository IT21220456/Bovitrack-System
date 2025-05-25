from fastapi import FastAPI, Form, HTTPException
import pickle
import numpy as np
from pydantic import BaseModel, field_validator
import joblib
import pandas as pd
from enum import Enum
from sklearn.preprocessing import StandardScaler
from typing import List

app = FastAPI()

# Load Models
env_con_model_path = "model/env_con_model.pkl"
with open(env_con_model_path, "rb") as f:
    env_con_model = pickle.load(f)

milk_pred_model = joblib.load("model/milk_quality_knn_model.joblib")

try:
    milk_yield_model = joblib.load("model/milk_yield_model.joblib")
except Exception as e:
    raise RuntimeError("Failed to load the model") from e

fec_model_path = 'model/Futu_env_con_rf.pkl'
with open(fec_model_path, 'rb') as file:
    fec_model = pickle.load(file)

# Classes and Functions
class PredictionRequest(BaseModel):
    features: list  

# Define enum for binary choices (0 for no, 1 for yes)
class BinaryOption(int, Enum):
    no = 0
    yes = 1

# Define enum for Colour options (values from 240 to 255)
class MilkColour(int, Enum):
    c240 = 240
    c241 = 241
    c242 = 242
    c243 = 243
    c244 = 244
    c245 = 245
    c246 = 246
    c247 = 247
    c248 = 248
    c249 = 249
    c250 = 250
    c251 = 251
    c252 = 252
    c253 = 253
    c254 = 254
    c255 = 255

# Function to create the input data dictionary
def create_input_data(pH: float, Temprature: float, taste: BinaryOption, 
                      odor: BinaryOption, fat: BinaryOption, 
                      turbidity: BinaryOption, colour: MilkColour) -> dict:
    return {
        "pH": pH,
        "Temprature": Temprature,
        "Taste": taste.value,
        "Odor": odor.value,
        "Fat ": fat.value,  # Note: the model was trained with 'Fat ' column name.
        "Turbidity": turbidity.value,
        "Colour": colour.value
    }

# The original Pydantic model is preserved in case it's needed for JSON endpoints
class MilkYieldInput(BaseModel):
    # Expect exactly 4 features: Length of Lactation, Days Dry (Days), Peak Yield (Kg), Days To Peak (Days)
    features: List[float]

    @field_validator("features") # Changed from @validator
    @classmethod # Add classmethod decorator
    def check_features_length(cls, v):
        if len(v) != 4:
            raise ValueError("features must have exactly 4 items")
        return v

# Endpoints
@app.get("/")  
def read_root():
    return {"message": "Welcome to the Bovi Track API!"}

@app.post("/env_con")  
def env_con(request: PredictionRequest):
    
    features = np.array(request.features).reshape(1, -1)  
    prediction = env_con_model.predict(features)  
    return {"prediction": prediction.tolist()}  

@app.post("/milk_pred")
async def milk_pred(
    pH: float = Form(...),
    Temprature: float = Form(...),
    Taste: BinaryOption = Form(...),
    Odor: BinaryOption = Form(...),
    Fat: BinaryOption = Form(...),
    Turbidity: BinaryOption = Form(...),
    Colour: MilkColour = Form(...)
):
    # Create input data using the dedicated function
    input_data = create_input_data(pH, Temprature, Taste, Odor, Fat, Turbidity, Colour)
    
    # Create a Pandas DataFrame from the input data
    input_df = pd.DataFrame([input_data])

    # Select the features used in training
    features = ['pH', 'Temprature', 'Taste', 'Odor', 'Fat ', 'Turbidity', 'Colour']
    X = input_df[features]

    # Apply standardization
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Make prediction using the KNN model
    knn_prediction = milk_pred_model.predict(X)[0]
    # Convert the numpy.int64 to a native int
    knn_prediction = int(knn_prediction)

    # Return the KNN prediction
    return knn_prediction
    # 0: Low (Bad)
    # 1: Medium (Moderate)
    # 2: High (Good)

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
    
@app.post("/future_env_condition")
async def future_env_condition(condition_today: int, tavg_today: float, hum_today: float):
    """
    Endpoint for getting predictions.
    """
    # Create a DataFrame with the input features
    input_data = pd.DataFrame({
        'condition_today': [condition_today],
        'tavg_today': [tavg_today],
        'hum_today': [hum_today]
    })

    # Make prediction using the loaded fec_model
    prediction = int(fec_model.predict(input_data)[0]) # Convert numpy.int64 to standard Python int

    if prediction == 0:
        return "Good"
    elif prediction == 1:
        return "Medium"
    elif prediction == 2:
        return "Poor"