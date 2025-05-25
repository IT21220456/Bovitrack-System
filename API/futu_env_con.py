from fastapi import FastAPI
import pickle
import pandas as pd

app = FastAPI()

# Load the trained fec_model
fec_model_path = 'model/Futu_env_con_rf.pkl'
with open(fec_model_path, 'rb') as file:
    fec_model = pickle.load(file)

@app.post("/predict")
async def predict(condition_today: str, tavg_today: float, hum_today: float):
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
    prediction = fec_model.predict(input_data)[0]
    # Convert numpy.int64 to standard Python int
    prediction_int = int(prediction)

    return {"prediction": prediction_int}