import cv2
from ultralytics import YOLO
import sys
import os
from contextlib import contextmanager

# Context manager to temporarily suppress stdout
@contextmanager
def suppress_stdout():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout

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

        # Initialize total counts and frame count for averaging
        total_cow_counts = {"eating": 0, "lying": 0, "standing": 0}
        frame_count = 0

        # Process the video stream frame by frame
        while cap.isOpened():
            ret, frame = cap.read()
            if ret:
                frame_count += 1 # Increment frame count
                # Suppress stdout during model prediction
                with suppress_stdout():
                    results = self.model(frame, verbose=False) # Add verbose=False to suppress YOLO's per-frame output
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

                # Accumulate counts
                total_cow_counts["eating"] += cow_counts["eating"]
                total_cow_counts["lying"] += cow_counts["lying"]
                total_cow_counts["standing"] += cow_counts["standing"]

                # Print the cow counts for the current frame (optional)
                # print(f"Frame {frame_count}: Lying: {cow_counts['lying']}, Eating: {cow_counts['eating']}, Standing: {cow_counts['standing']}")
            else:
                break

        cap.release()

        # Calculate and print the average counts
        if frame_count > 0:
            avg_lying = total_cow_counts['lying'] / frame_count
            avg_eating = total_cow_counts['eating'] / frame_count
            avg_standing = total_cow_counts['standing'] / frame_count
            print("\n--- Final Average Counts ---")
            print(f"Average Lying: {avg_lying:.2f}")
            print(f"Average Eating: {avg_eating:.2f}")
            print(f"Average Standing: {avg_standing:.2f}")
        else:
            print("No frames processed.")
