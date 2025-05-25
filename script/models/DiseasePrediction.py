import pandas as pd
import joblib

class DiseasePredictionModel:
    def __init__(self):
        self.label_mapping = {
            0: 'Circulatory Shock',
            1: 'Healthy',
            2: 'Heat Stress',
            3: 'Mastitis',
            4: 'Milk Fever',
            5: 'Respiratory Disease'
        }
        # self.loaded_model = joblib.load('MLModels\\cows_diseases_detection_model_XGB.pkl')
        self.loaded_model = joblib.load('MLModels\\knn_cow_disease_model.joblib')



    def predict_disease(self, temperature, blood_pressure, blood_oxygen):
        user_input = pd.DataFrame([[temperature, blood_pressure, blood_oxygen]],
                                  columns=['temperature', 'blood_pressure', 'blood_oxygen'])
        prediction = self.loaded_model.predict(user_input)
        return self.label_mapping[prediction[0]]
