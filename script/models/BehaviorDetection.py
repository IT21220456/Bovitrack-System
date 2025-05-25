import cv2
from ultralytics import YOLO

class BehaviorDetectionModel:
    def __init__(self):
        # Provide the local path to your video file here
        # self.video_path = 'data/01.jpg'
        self.video_path = 'data\\01.mp4'  # **********Replace video path ***********s
        self.model = YOLO('MLModels\\best.pt')

    def predict_behavior(self):
        # Open the video stream using OpenCV
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            print("Error opening video file")
            return
        
        # Process the video stream frame by frame
        while cap.isOpened():
            ret, frame = cap.read()
            if ret:
                results = self.model(frame)
                cow_counts = {"eating": 0, "lying": 0, "standing": 0}

                # Loop through the detected boxes and classify them
                for result in results:
                    if hasattr(result, 'boxes'):
                        for box in result.boxes:
                            class_id = int(box.cls[0])
                            if class_id == 0:  # Adjust according to your model's class IDs
                                cow_counts["eating"] += 1
                            elif class_id == 1:
                                cow_counts["lying"] += 1
                            elif class_id == 2:
                                cow_counts["standing"] += 1

                # Print the cow counts
                print(f"Lying: {cow_counts['lying']}, Eating: {cow_counts['eating']}, Standing: {cow_counts['standing']}")
            else:
                break

        cap.release()
