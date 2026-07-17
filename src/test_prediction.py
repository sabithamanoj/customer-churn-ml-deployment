"""
test_prediction.py

End-to-end test:
Raw customer input
        |
        ▼
Preprocessing
        |
        ▼
Model prediction
"""


import logging

from preprocess import preprocess_input
from predict import predict_churn


# ---------------------------------------------------------
# Configure logging
# ---------------------------------------------------------

logging.basicConfig(
    filename='test_prediction.log',
    level=logging.INFO, filemode='w', format='%(message)s'
)
logging.getLogger().addHandler(logging.StreamHandler())

# ---------------------------------------------------------
# Sample customer
# ---------------------------------------------------------

sample_customer = {

    "gender": "Female",
    "SeniorCitizen": 0,
    "Partner": "Yes",
    "Dependents": "No",
    "tenure": 12,
    "PhoneService": "Yes",
    "InternetService": "Fiber optic",
    "Contract": "Month-to-month",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 80,
    "TotalCharges": 960

}


# ---------------------------------------------------------
# Preprocess input
# ---------------------------------------------------------

processed_input = preprocess_input(
    sample_customer
)


logging.info(
    "Processed input shape: %s",
    processed_input.shape
)


# ---------------------------------------------------------
# Generate prediction
# ---------------------------------------------------------

result = predict_churn(
    processed_input
)


# ---------------------------------------------------------
# Log prediction result
# ---------------------------------------------------------

logging.info(
    "Prediction result: %s",
    result
)


logging.info(
    "Final prediction output: %s",
    result
)