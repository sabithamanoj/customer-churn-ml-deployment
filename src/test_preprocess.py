from preprocess import preprocess_input


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


processed = preprocess_input(
    sample_customer
)


print(processed)

print(
    processed.shape
)