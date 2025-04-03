# Fraud Detection App

This project is a simple Streamlit application for detecting fraudulent financial transactions.

## Project Contents:
- `app.py` — the main Streamlit app that allows users to upload a CSV file and detect fraud
- `fraud_model.pkl` — the saved machine learning model used for predictions

## How to Run the App:

1. Install the required packages:
```bash
pip install streamlit pandas scikit-learn
# Run the Steamlit app:
Streamlit run app.py
Upload a CSV file with transaction data and get prediction on which transaction may be fraudlet

## Features:
Upload and analyze custom dataset
Predict fraud using a trained model
View result directly in browser
