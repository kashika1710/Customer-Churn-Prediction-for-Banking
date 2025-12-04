# Bank Customer Churn Prediction & Dashboard

## Overview
This project predicts which banking customers are likely to leave the bank (churn)
using machine learning models and presents analysis through an interactive dashboard.

## Business Problem
Customer churn results in revenue loss and higher acquisition costs. Predicting churn helps
banks proactively retain at-risk customers through targeted offers and service interventions.

## Dataset
The dataset contains demographic, behavioral, and banking information including:
- Credit Score
- Age and Gender
- Geography
- Balance and Salary
- Complaints and Satisfaction Score
- Loyalty Points and Card Type

## Features & Methods
- Data cleaning and preprocessing
- Feature engineering and customer segmentation
- Model training (Logistic Regression, Random Forest, Gradient Boosting)
- Model evaluation (ROC-AUC, Confusion Matrix)
- Explainable AI using SHAP
- Model deployment with Streamlit

## Dashboard Preview
The Streamlit app allows you to:
- Enter customer details and predict churn probability
- View customer characteristics instantly
- Make business-driven decisions

## Installation & Usage

```bash
pip install -r requirements.txt
streamlit run app.py
