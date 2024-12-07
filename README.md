# BoviTrack

A brief description of our project.

## Table of Contents
1. Functions
    -  Function 1
        -  Model 1 : Behavioral Detection Model
    -  Function 2
        -  Model 1 : Video_Quality_Prediction_Model
        -  Algorithm 1 : Dynamic Video Resolution Adjustment Algorithm
        -  Algorithm 2 : Frame Rate Reduction Algorithm for Live Streaming
    -  Function 3
        -  Model 1 : Health Monitoring Model
    -  Function 4
        -  Model 1 : environmental conditions detection model
        -  Model 2 : water quality detection model
2. API
3. How to Setup

---

## 1. Functions

### Function 1: 
#### Model 1: 

- **Dataset (Drive or GitHub URL)**:  https://universe.roboflow.com/different-4qull/cows_movement_and_behaviours-cehrf/dataset/2#
- **Final Code (Folder URL)**: 
- **Use Technologies and Model**: 
- **Model Label**: 
- **Model Features**: 
- **Model (GitHub or Drive URL)**:
- **Tokenizer (GitHub or Drive URL)**:
- **Scaler (GitHub or Drive URL)**: 
- **Accuracy**:
- **How to Load and Get Prediction for One Input**:
    ```python
    import joblib

    model = joblib.load('models/linear_regression.pkl')
    scaler = joblib.load('scalers/scaler.pkl')

    input_data = [[2500, 4, 'Suburban']]
    scaled_data = scaler.transform(input_data)
    prediction = model.predict(scaled_data)
    print("Predicted Price:", prediction)
    ```

#### Model 2: 

- **Dataset (Drive or GitHub URL)**: 
- **Final Code (Folder URL)**: 
- **Use Technologies and Model**: 
- **Model Label**: 
- **Model Features**: 
- **Model (GitHub or Drive URL)**:
- **Tokenizer (GitHub or Drive URL)**:
- **Scaler (GitHub or Drive URL)**: 
- **Accuracy**:
- **How to Load and Get Prediction for One Input**:
    ```python
    import joblib

    model = joblib.load('models/linear_regression.pkl')
    scaler = joblib.load('scalers/scaler.pkl')

    input_data = [[2500, 4, 'Suburban']]
    scaled_data = scaler.transform(input_data)
    prediction = model.predict(scaled_data)
    print("Predicted Price:", prediction)
    ```

### Function 2:
#### Model 2:

- **Dataset (Drive or GitHub URL)**: 
- **Final Code (Folder URL)**: 
- **Use Technologies and Model**: 
- **Model Label**: 
- **Model Features**: 
- **Model (GitHub or Drive URL)**:
- **Tokenizer (GitHub or Drive URL)**:
- **Scaler (GitHub or Drive URL)**: 
- **Accuracy**:
- **How to Load and Get Prediction for One Input**:
    ```python
    import joblib

    model = joblib.load('models/decision_tree.pkl')
    input_data = [[5.1, 3.5, 1.4, 0.2]]
    prediction = model.predict(input_data)
    print("Predicted Species:", prediction)
    ```
---

## 2. API

- **Use Technology**: Flask, Swagger
- **API Folder (Drive or GitHub URL)**: 
- **API Folder Screenshot**: 
    
- **API Testing Swagger Screenshots for All Endpoints**:
    

---

## 3. How to Stup

### Pre-requisites

### Steps
1. Clone the repository:
    ```bash
    git clone https://github.com/username/repo.git
    ```
2. Navigate to the project directory:
    ```bash
    cd project-directory
    ```
3. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
4. Run the application:
    ```bash
    fastapi dev main.py
    ```

## 4. Others
- Contact: Your Name 
---

Feel free to modify this template to better fit your project's specifics and structure!
