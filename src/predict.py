
"""
predict.py

This module handles model loading and prediction logic
for the customer churn prediction application.

Responsibilities:
1. Load the trained ML model.
2. Load preprocessing artifacts.
3. Generate churn predictions.
4. Return prediction results.
"""


# Import required libraries
import os
import joblib
import pandas as pd


# ---------------------------------------------------------
# 1. Define paths to saved model artifacts
# ---------------------------------------------------------

# Get the root directory of the project
# Current file:
# customer-churn-ml-deployment/src/predict.py
#
# Move two levels up to:
# customer-churn-ml-deployment/

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)


# Define the models directory path

MODEL_DIR = os.path.join(
    BASE_DIR,
    "models"
)


# ---------------------------------------------------------
# 2. Load trained model and supporting files
# ---------------------------------------------------------

# Load the trained machine learning model
# This model was trained in the Google Colab notebook
# and saved using joblib.

MODEL_PATH = os.path.join(
    MODEL_DIR,
    "churn_model.pkl"
)

model = joblib.load(MODEL_PATH)


# Load feature names used during training.
#
# This is extremely important because the model expects
# exactly the same columns and column order during inference.

FEATURE_PATH = os.path.join(
    MODEL_DIR,
    "feature_names.pkl"
)

feature_names = joblib.load(FEATURE_PATH)


# Load scaler.
#
# Note:
# Gradient Boosting / Random Forest do not require scaling.
# However, we save and load it because:
# - it was part of the training pipeline
# - it keeps the deployment pipeline flexible
# - it can be used if the model changes later

SCALER_PATH = os.path.join(
    MODEL_DIR,
    "scaler.pkl"
)

scaler = joblib.load(SCALER_PATH)


# ---------------------------------------------------------
# 3. Prediction Function
# ---------------------------------------------------------

def predict_churn(input_data):
    """
    Predict customer churn probability.

    Parameters
    ----------
    input_data : pandas.DataFrame
        Preprocessed customer data with the same
        features used during training.

    Returns
    -------
    dict
        Prediction label and probability.
    """


    # -----------------------------------------------------
    # Ensure input columns match training columns
    # -----------------------------------------------------

    # Reorder columns according to training order.
    #
    # Machine learning models do not understand column names.
    # They use the position of values.
    #
    # Example:
    # Training:
    # [age, tenure, monthly_charge]
    #
    # Prediction:
    # [monthly_charge, age, tenure]
    #
    # would produce incorrect predictions.

    input_data = input_data[
        feature_names
    ]


    # -----------------------------------------------------
    # Generate prediction
    # -----------------------------------------------------

    # Predict class:
    # 0 --> No Churn
    # 1 --> Churn

    prediction = model.predict(
        input_data
    )[0]


    # Predict probability of churn.
    #
    # predict_proba returns:
    #
    # [[probability_class_0,
    #   probability_class_1]]
    #
    # We select class 1 probability.

    churn_probability = model.predict_proba(
        input_data
    )[0][1]


    # -----------------------------------------------------
    # Convert numerical prediction into readable output
    # -----------------------------------------------------

    if prediction == 1:

        prediction_label = "Churn"

    else:

        prediction_label = "No Churn"


    # -----------------------------------------------------
    # Return API-friendly response
    # -----------------------------------------------------

    result = {

        "prediction": prediction_label,

        "churn_probability": round(
            float(churn_probability),
            4
        )

    }


    return result
