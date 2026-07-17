"""
preprocess.py

This module handles preprocessing of incoming customer data
before sending it to the trained machine learning model.

Responsibilities:
1. Convert input JSON into DataFrame.
2. Apply categorical encoding.
3. Match training feature structure.
4. Return model-ready data.
"""


# Import libraries

import pandas as pd
import os
import joblib


# ---------------------------------------------------------
# 1. Load training feature information
# ---------------------------------------------------------

# Find project root directory

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)


# Path to models directory

MODEL_DIR = os.path.join(
    BASE_DIR,
    "models"
)


# Load feature names created during training

FEATURE_PATH = os.path.join(
    MODEL_DIR,
    "feature_names.pkl"
)


feature_names = joblib.load(
    FEATURE_PATH
)



# ---------------------------------------------------------
# 2. Preprocessing Function
# ---------------------------------------------------------

def preprocess_input(customer_data):
    """
    Convert raw customer input into
    model-compatible format.

    Parameters
    ----------
    customer_data : dict
        Raw customer information received
        from API request.

    Returns
    -------
    pandas.DataFrame
        Processed input ready for prediction.
    """


    # -----------------------------------------------------
    # Convert dictionary to DataFrame
    # -----------------------------------------------------

    # Example input:
    #
    # {
    # "gender": "Female",
    # "tenure": 12,
    # "Contract": "Month-to-month"
    # }
    #
    # becomes:
    #
    # DataFrame with one row


    df = pd.DataFrame(
        [customer_data]
    )


    # -----------------------------------------------------
    # Convert TotalCharges to numeric
    # -----------------------------------------------------

    # During training we converted TotalCharges
    # using:
    #
    # pd.to_numeric(errors="coerce")
    #
    # We repeat the same operation here.

    if "TotalCharges" in df.columns:

        df["TotalCharges"] = pd.to_numeric(
            df["TotalCharges"],
            errors="coerce"
        )


    # -----------------------------------------------------
    # One-hot encode categorical variables
    # -----------------------------------------------------

    # Convert categorical columns into numerical columns.



    df = pd.get_dummies(
        df,
        drop_first=True
    )


    # -----------------------------------------------------
    # Match training features
    # -----------------------------------------------------

    # Add missing columns.
    #
    # Example:
    #
    # Training data had:
    # InternetService_Fiber optic
    #
    # But new customer data does not contain it.
    #
    # Add it with value 0.

    for column in feature_names:

        if column not in df.columns:

            df[column] = 0



    # -----------------------------------------------------
    # Remove unexpected columns
    # -----------------------------------------------------

    # Keep only features used during training.

    df = df[
        feature_names
    ]


    # -----------------------------------------------------
    # Return processed data
    # -----------------------------------------------------

    return df