import numpy as np
import pickle

class ConditionPredictionModel:
    def __init__(self):
        with open('MLModels\\env_con_model.pkl', 'rb') as file:
            self.loaded_model = pickle.load(file)

    def predict_condition(self, tavg, hum):
        # Create the input array for the model
        input_array = np.array([[tavg, hum]])
        # Get the model's prediction
        prediction = self.loaded_model.predict(input_array)
        
        # Map the prediction to corresponding label
        if prediction[0] == 0:
            return "Good"
        elif prediction[0] == 1:
            return "Medium"
        elif prediction[0] == 2:
            return "Poor"
