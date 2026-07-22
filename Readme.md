# Customer Churn Prediction — End-to-End Machine Learning Deployment

An end-to-end Machine Learning project that predicts customer churn using a supervised learning model and deploys it as a REST API using Flask and Docker.

This project demonstrates the complete Machine Learning workflow, from data preprocessing and model training to API development, testing, and containerized deployment.

---

## 📌 Project Overview

This project covers the complete Machine Learning lifecycle:

- Data preprocessing and feature engineering
- Exploratory Data Analysis (EDA)
- Model training and evaluation
- Model serialization using Joblib
- REST API development with Flask
- API testing using Postman
- Docker containerization
- End-to-end deployment

---

## 📂 Project Structure

```text
customer-churn-ml-deployment/
│
├── Dockerfile
├── .dockerignore
├── README.md
├── requirements.txt
│
├── models/
│   ├── churn_model.pkl
│   ├── feature_names.pkl
│   └── scaler.pkl
│
├── notebooks/
│   └── Customer_Churn_Model.ipynb
│
├── src/
│   ├── app.py
│   ├── predict.py
│   ├── preprocess.py
│   └── customer_churn_api.log
│
├── api_tests/
│
├── postman/
│
└── reports/
```

---

## 🚀 Machine Learning Workflow

```text
Collect Data
      │
      ▼
Data Cleaning
      │
      ▼
Exploratory Data Analysis
      │
      ▼
Feature Engineering
      │
      ▼
Train/Test Split
      │
      ▼
Model Training
      │
      ▼
Model Evaluation
      │
      ▼
Save Model Artifacts
      │
      ▼
Flask REST API
      │
      ▼
Postman API Testing
      │
      ▼
Docker Deployment
```

---

## 🛠 Technologies Used

### Programming

- Python

### Machine Learning

- Scikit-learn
- Pandas
- NumPy
- Joblib

### API Development

- Flask

### API Testing

- Postman

### Containerization

- Docker

---

## 🌐 REST API

### Health Check

**Endpoint**

```http
GET /
```

**Response**

```json
{
    "message": "Customer Churn Prediction API is running."
}
```

---

### Predict Customer Churn

**Endpoint**

```http
POST /predict
```

**Example Request**

```json
{
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
```

**Example Response**

```json
{
    "prediction": "No Churn",
    "churn_probability": 0.2095
}
```

---

## 🐳 Docker Deployment

The application has been containerized using Docker.

### Build the Docker image

```bash
docker build -t customer-churn-api .
```

### Run the container

```bash
docker run -p 5000:5000 customer-churn-api
```

After the container starts, the API will be available at:

```text
http://127.0.0.1:5000
```

---

## 📖 Deployment Guide

A detailed deployment guide is included in this repository.

It covers:

- Project architecture
- Flask API implementation
- Input preprocessing
- Model loading
- Prediction pipeline
- Postman API testing
- Docker image creation
- Running Docker containers
- Troubleshooting

Please refer to the deployment documentation for the complete step-by-step guide.

**Readme_deployment.ipynb** 

---

## ⭐ Project Highlights

- End-to-end Machine Learning workflow
- Clean and modular project structure
- Reusable preprocessing pipeline
- Feature consistency between training and inference
- REST API using Flask
- Structured logging
- Exception handling
- Dockerized deployment
- Postman testing

---

## 🔮 Future Improvements

- Production deployment using Gunicorn
- Reverse proxy with Nginx
- CI/CD using GitHub Actions
- Cloud deployment (Azure, AWS, or Google Cloud)
- Automated model monitoring
- API authentication
- Continuous model retraining


---

## 🙏 Acknowledgements

This project was developed as a practical demonstration of deploying a Machine Learning model as a scalable REST API using modern Python development and containerization practices.