"""
app.py

Flask application for Customer Churn Prediction.
"""

# ---------------------------------------------------------
# Import libraries
# ---------------------------------------------------------

import logging
import os

from flask import Flask, jsonify, request

from preprocess import preprocess_input
from predict import predict_churn


# ---------------------------------------------------------
# Configure logging safely
# ---------------------------------------------------------

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

LOG_FILE = os.path.join(
    BASE_DIR,
    "customer_churn_api.log"
)


logger = logging.getLogger("customer_churn_api")

logger.setLevel(logging.INFO)


# Avoid adding multiple handlers
if not logger.handlers:

    file_handler = logging.FileHandler(
        LOG_FILE,
        mode="a"
    )

    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(message)s"
    )

    file_handler.setFormatter(
        formatter
    )

    logger.addHandler(
        file_handler
    )


# ---------------------------------------------------------
# Create Flask application
# ---------------------------------------------------------

app = Flask(__name__)


logger.info(
    "Customer Churn Prediction API initialized."
)


# ---------------------------------------------------------
# Health endpoint
# ---------------------------------------------------------

@app.route("/", methods=["GET"])
def home():

    logger.info(
        "Health check endpoint called."
    )

    return jsonify({

        "message":
        "Customer Churn Prediction API is running."

    })


# ---------------------------------------------------------
# Prediction endpoint
# ---------------------------------------------------------

@app.route("/predict", methods=["POST"])
def predict():

    try:

        customer_data = request.get_json()


        logger.info(
            "Received input: %s",
            customer_data
        )


        processed_input = preprocess_input(
            customer_data
        )


        logger.info(
            "Preprocessing completed. Shape: %s",
            processed_input.shape
        )


        result = predict_churn(
            processed_input
        )


        logger.info(
            "Prediction result: %s",
            result
        )


        return jsonify(result)


    except Exception as e:

        logger.exception(
            "Prediction failed."
        )

        return jsonify({

            "error": str(e)

        }), 500



# ---------------------------------------------------------
# Start application
# ---------------------------------------------------------

if __name__ == "__main__":

    print("Starting Customer Churn API...")

    app.run(
        host="0.0.0.0",
        port=5000,
        debug=False
    )