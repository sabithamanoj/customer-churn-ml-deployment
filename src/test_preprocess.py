"""
test_preprocess.py

Tests the preprocessing pipeline by:
1. Sending sample customer data.
2. Applying preprocessing.
3. Logging the processed output.
4. Checking feature shape and columns.
"""


import logging

from preprocess import preprocess_input


# ---------------------------------------------------------
# Configure logging
# ---------------------------------------------------------

logging.basicConfig(
    filename='test_preprocess.log',
    level=logging.INFO, filemode='w', format='%(message)s'
)
logging.getLogger().addHandler(logging.StreamHandler())



# ---------------------------------------------------------
# Sample customer input
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
# Apply preprocessing
# ---------------------------------------------------------

processed = preprocess_input(
    sample_customer
)


# ---------------------------------------------------------
# Log processed dataframe
# ---------------------------------------------------------

logging.info(
    "Processed input dataframe:"
)

logging.info(
    "\n%s",
    processed.to_string()
)


# ---------------------------------------------------------
# Log dataframe shape
# ---------------------------------------------------------

logging.info(
    "Processed input shape: %s",
    processed.shape
)


# ---------------------------------------------------------
# Log feature columns
# ---------------------------------------------------------

logging.info(
    "Processed feature columns:"
)


for index, column in enumerate(processed.columns):

    logging.info(
        "%d : %s",
        index,
        column
    )