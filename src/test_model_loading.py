import joblib
import os


MODEL_PATH = "../models/churn_model.pkl"

model = joblib.load(MODEL_PATH)

print("Model loaded successfully!")
print(type(model))