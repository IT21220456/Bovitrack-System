from fastapi import FastAPI
import pickle
import numpy as np
from pydantic import BaseModel


app = FastAPI()


env_con_model_path = "models/env_con_model.pkl"
with open(env_con_model_path, "rb") as f:
    env_con_model = pickle.load(f)


class PredictionRequest(BaseModel):
    features: list  

@app.get("/")  
def read_root():
    return {"message": "Welcome to the Bovi Track API!"}

@app.post("/env_con")  
def env_con(request: PredictionRequest):
    
    features = np.array(request.features).reshape(1, -1)  
    prediction = env_con_model.predict(features)  
    return {"prediction": prediction.tolist()}  


