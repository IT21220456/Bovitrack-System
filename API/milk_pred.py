import joblib
import pandas as pd
from fastapi import FastAPI, Form
from enum import Enum
from sklearn.preprocessing import StandardScaler

app = FastAPI()

# Load the KNN model
milk_pred_model = joblib.load("model/milk_quality_svm_model.joblib")

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

# Define the prediction endpoint with form inputs and dropdown options
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