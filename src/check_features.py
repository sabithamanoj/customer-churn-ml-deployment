"""
check_features.py

Utility script to inspect the features
used during model training.
"""

import logging
import joblib


# ---------------------------------------------------------
# Configure logging
# ---------------------------------------------------------
logging.basicConfig(
    filename='check_features.log',
    level=logging.INFO, filemode='w', format='%(message)s'
)
logging.getLogger().addHandler(logging.StreamHandler())

# ---------------------------------------------------------
# Load saved training features
# ---------------------------------------------------------

feature_names = joblib.load(
    "../models/feature_names.pkl"
)


# ---------------------------------------------------------
# Log number of features
# ---------------------------------------------------------

logging.info(
    "Number of features used by model: %d",
    len(feature_names)
)


# ---------------------------------------------------------
# Log feature names
# ---------------------------------------------------------

logging.info(
    "Feature list:"
)


for i, feature in enumerate(feature_names):

    logging.info(
        "%d : %s",
        i,
        feature
    )