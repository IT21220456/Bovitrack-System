# Inputs & Outputs



## Milk Yield

**Inputs:**

| Feature | Type | Description | Limits (Approximate) |
|---|---|---|---|
| Length of Lactation | Numeric | Total duration of milk production (days) | 100 - 500 days |
| Days Dry | Numeric | Non-lactating period (days) | 30 - 90 days |
| Peak Yield | Numeric | Highest daily milk production (Kg) | 10 - 50+ Kg |
| Days To Peak | Numeric | Time to reach peak yield (days) | 30 - 100 days |

**Outputs:**

| Output | Type | Description | Limits |
|---|---|---|---|
| Total Milk Yield | Numeric | Total milk produced during lactation (Kg) | Dataset dependent, potentially thousands of Kg |
| Predicted Class | Categorical | High (1) or Low (0) yield based on threshold | Determined by threshold in code |

**Model Limits:**

*   **Linearity:** Assumes a linear relationship between features and target. May not capture complex interactions.
*   **Data Dependency:** Model is specific to its training data. Predictions might be less accurate for data outside the training range.
*   **Feature Importance:** Accuracy is tied to chosen features. Adding/removing features can change performance.
*   **Accuracy:** Evaluate with appropriate metrics (e.g., R-squared) for overall prediction accuracy.

## Milk Quality

## Milk Quality Prediction Model: Inputs, Outputs, and Limits

This document outlines the inputs, outputs, and limitations of the trained machine learning models for milk quality prediction. Two models were trained and saved: Support Vector Machine (SVM) and K-Nearest Neighbors (KNN).

### Inputs

| Feature | Type | Description | Range/Limits |
|---|---|---|---|
| pH | Float | The pH level of the milk | Typically between 6.5 and 6.8 for cow milk |
| Temperature | Float | The temperature of the milk in Celsius | Typically between 2-40 degrees Celsius|
| Taste | Integer (0 or 1) | A binary feature indicating whether the taste of the milk is satisfactory (1) or not (0) | 0 or 1 |
| Odor | Integer (0 or 1) | A binary feature indicating whether the odor of the milk is satisfactory (1) or not (0) | 0 or 1 |
| Fat  | Integer (0 or 1)| The fat content of the milk. Optimal conditions represented by 1, otherwise 0 | 0 or 1 |
| Turbidity | Integer (0 or 1) | A measure of the milk's cloudiness or haziness. Optimal conditions represented by 1, otherwise 0 | 0 or 1 |
| Colour | Integer | The color of the milk (numerical representation) | Typically ranges from white to slightly yellowish |


**Note:** The input features are standardized before being fed into the model using a `StandardScaler` fitted on the training data.

### Outputs

| Output | Type | Description |
|---|---|---|
| Grade | Integer | The predicted quality grade of the milk |


The output `Grade` represents the predicted quality of the milk and can take on one of the following values:

- 0: Low (Bad)
- 1: Medium (Moderate)
- 2: High (Good)


### Model Limits

- **Data Dependency:** The model's performance is highly dependent on the quality and representativeness of the training data. If the training data is biased or incomplete, the model may not generalize well to new, unseen data.
- **Limited Features:** The model considers only a limited set of features. Other factors that could influence milk quality, such as the presence of bacteria or antibiotics, are not included in the model.
- **Accuracy:** While the models achieved relatively high accuracy on the testing data, there is always a possibility of misclassification. The accuracy metrics for each model can be found in the model evaluation section of the notebook.
- **Interpretability:** The SVM model with a linear kernel offers some level of interpretability by examining the feature weights. However, more complex kernels can make interpretation challenging. The KNN model's decisions are based on the nearest neighbors, which can be harder to interpret directly.