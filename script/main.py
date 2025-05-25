from models.BehaviorDetection import BehaviorDetectionModel
from models.BehaviorDetectionAverage import BehaviorDetectionModel as BehaviorDetectionModelAvg
from models.DiseasePrediction import DiseasePredictionModel
from models.WaterSuitabilityPrediction import SuitabilityPredictionModel
from models.EnvConditionPrediction import ConditionPredictionModel

class PredictionController:
    def __init__(self):
        self.model1 = BehaviorDetectionModel()  # For behavior detection
        self.model2 = BehaviorDetectionModelAvg()  # For average behavior detection
        self.model3 = DiseasePredictionModel()  # For disease prediction
        self.model4 = SuitabilityPredictionModel()  # For water suitability prediction
        self.model5 = ConditionPredictionModel()  # For enviroment condition prediction

    def run_model1(self):
        self.model1.predict_behavior()
        input("\nPress Enter to return to the main menu...")

    def run_model2(self):
        self.model2.predict_behavior()
        input("\nPress Enter to return to the main menu...")

    def run_model3(self):
        while True:
            try:
                temperature = float(input("Enter the cow's temperature: "))
                blood_pressure = float(input("Enter the cow's blood pressure: "))
                blood_oxygen = float(input("Enter the cow's blood oxygen level: "))
                result = self.model3.predict_disease(temperature, blood_pressure, blood_oxygen)
                print(f"The predicted disease is: {result}")
            except ValueError:
                print("Invalid input. Please enter numeric values.")
                continue

            choice = input("\nDo you want to try again? (y/n): ").lower()
            if choice != 'y':
                break

    def run_model4(self):
        self.model4.predict_suitability()
        input("\nPress Enter to return to the main menu...")

    def run_model5(self):
        while True:
            try:
                tavg = float(input("Enter the average temperature (tavg): "))
                hum = float(input("Enter the humidity (hum): "))
                result = self.model5.predict_condition(tavg, hum)
                print(f"Predicted Condition: {result}")
            except ValueError:
                print("Invalid input. Please enter numeric values.")
                continue

            choice = input("\nDo you want to try again? (y/n): ").lower()
            if choice != 'y':
                break

if __name__ == '__main__':
    controller = PredictionController()
    
    while True:
        print("\nSelect Model to Run:")
        print("1. Behavior Detection (YOLO)")
        print("2. Behavior Detection Average (YOLO)")
        print("3. Disease Prediction (XGBoost)")
        print("4. Water Suitability Prediction (XGBoost)")
        print("5. Environment Condition Prediction (XGBoost)")
        print("6. Exit")

        choice = input("\nEnter your choice (1-5): ")

        if choice == '1':
            controller.run_model1()
        elif choice == '2':
            controller.run_model2()
        elif choice == '3':
            controller.run_model3()
        elif choice == '4':
            controller.run_model4()
        elif choice == '5':
            controller.run_model5()
        elif choice == '6':
            print("Exiting... Goodbye!")
            break
        else:
            print("Invalid choice. Please enter a number between 1 and 6.")
