# BoviTrack

A brief description of our project.

## Project Details
Project ID - 24-25J-008
Project Member Details

Member 1

    - Member Name - Samaranayaka S.G.H.V
    - Member ID - IT21213908
    - Member email - it21213908@my.sliit.lk


Member 2

    - Member Name - Wijesekara S.P.
    - Member ID - IT21220456
    - Member email - it21220456@my.sliit.lk


Member 3

    - Member Name - Tharuka G.A.K
    - Member ID - IT21221514
    - Member email - it21221514@my.sliit.lk


Member 4

    - Member Name - Fernando W.W.S.T
    - Member ID - IT21307362
    - Member email - it21307362@my.sliit.lk


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
- **Final Code (Folder URL)**: https://github.com/IT21220456/Bovitrack-System/blob/main/Function%201/Behavior_monitoring_model_testing_with_bit_rate_Controling.ipynb
- **Use Technologies and Model**: YOLO
- **Model Target**: Behavior
- **Model Features**: Eating, Lying, Standing


### Function 2:
#### Model 1:

- **Dataset (Drive or GitHub URL)**: https://github.com/IT21220456/Bovitrack-System/blob/main/Function%202/video_quality_prediction_dataset.csv
- **Final Code (Folder URL)**: https://github.com/IT21220456/Bovitrack-System/blob/main/Function%202/Video_Quality_Prediction_Model-with-CMs.ipynb
- **Use Technologies and Model**: Decision Tree classifier 
- **Model Target**: Predicted_Video_Quality
- **Model Features**: Connection_Speed (Mbps), Video_Resolution (p), Video_FPS, Buffering_Rate (s)
- **Accuracy**: 0.9995

#### Algorithm 1:

- **Final Code (Folder URL)**: https://github.com/IT21220456/Bovitrack-System/blob/main/Function%201/Behavior_monitoring_model_testing_with_bit_rate_Controling.ipynb
- **perpose**: dynamically adjusts video resolution based on the available internet speed to optimize video processing for the YOLO model.

#### Algorithm 2:

- **Final Code (Folder URL)**: https://github.com/IT21220456/Bovitrack-System/blob/main/Function%202/Frame_rate_reducer_algorithm%20(1).ipynb
- **perpose**: intelligently reduces the frame rate of live-streamed videos to optimize bandwidth usage while preserving essential visual information. It balances the trade-off between smooth streaming and data transmission efficiency.


### Function 3:
#### Model 1:

- **Dataset (Drive or GitHub URL)**: https://github.com/IT21220456/Bovitrack-System/blob/main/Function%203/synthetic_cow_health_data%20(1).csv
- **Final Code (Folder URL)**: https://github.com/IT21220456/Bovitrack-System/blob/main/Function%203/Cows_Diseases_Detection_Model%20%20(1).ipynb
- **Use Technologies and Model**: XGBoost classifier
- **Model Target**: disease
- **Model Features**: temperature, blood_pressure, blood_oxygen	
- **Accuracy**: 0.94


### Function 4:
#### Model 1:

- **Dataset (Drive or GitHub URL)**: https://github.com/IT21220456/Bovitrack-System/blob/main/Function%204/environmental%20conditions%20detection%20model/Nuwaraeliya_c_data%20(1).csv
- **Final Code (Folder URL)**: https://github.com/IT21220456/Bovitrack-System/blob/main/Function%204/environmental%20conditions%20detection%20model/environmental%20conditions%20detection%20model%20V1%20(1).ipynb
- **Use Technologies and Model**: XGBoost classifier
- **Model Target**: Condition
- **Model Features**: temperature , humidity	
- **Accuracy**: 0.996

  #### Model 2:

- **Dataset (Drive or GitHub URL)**: https://github.com/IT21220456/Bovitrack-System/blob/main/Function%204/water%20quality%20detection%20model/cow_water_quality_dataset_with_patterns_DR.csv
- **Final Code (Folder URL)**: https://github.com/IT21220456/Bovitrack-System/blob/main/Function%204/water%20quality%20detection%20model/water_Quality_prediction_model-DR_Final.ipynb
- **Use Technologies and Model**: XGBoost classifier 
- **Model Target**: suitable
- **Model Features**: pH,TDS,Temperature
- **Accuracy**: 0.997

  
  
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
