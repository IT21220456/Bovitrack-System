import pandas as pd
import pickle

class SuitabilityPredictionModel:
    def __init__(self):
        self.model = self.load_model()
        # Update to the actual feature names used during model training
        self.feature_columns = ['pH', 'Total_Dissolved_Solids', 'Temperature']

    def load_model(self):
        with open('MLModels\\water_quality.pkl', 'rb') as file:
            model = pickle.load(file)
        #print("Loaded XGBoost model")
        return model

    def predict_suitability(self):
        user_data = {}
        for column in self.feature_columns:
            user_data[column] = float(input(f"Enter {column}: "))
        
        user_input_df = pd.DataFrame([user_data])
        user_input_df = user_input_df[self.feature_columns]
        
        prediction = self.model.predict(user_input_df)
        suitability = 'Suitable' if prediction[0] == 0 else 'Unsuitable'
        print(f"Predicted Suitability: {suitability}")
