import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# ========= CONFIG =========
MODEL_PATH = "C://Users//Dell-Pc//Desktop//Data Banking//best_churn_model_pipeline.pkl"  

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        st.error(f"Model file not found at: {MODEL_PATH}")
        st.stop()
    model = joblib.load(MODEL_PATH)
    return model

st.set_page_config(page_title="Bank Churn Prediction", layout="centered")

st.title("💳 Bank Customer Churn Prediction")
st.write("Enter customer details to predict if they are likely to **churn (exit)** the bank.")

model = load_model()

# ----- Sidebar inputs -----
st.sidebar.header("Customer Details")

def get_user_input():
    CreditScore = st.sidebar.slider("Credit Score", 300, 900, 650)
    Geography = st.sidebar.selectbox("Geography", ["France", "Germany", "Spain"])
    Gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
    Age = st.sidebar.slider("Age", 18, 90, 40)
    Tenure = st.sidebar.slider("Tenure (Years with bank)", 0, 10, 5)
    Balance = st.sidebar.number_input("Balance", min_value=0.0, max_value=300000.0, value=50000.0, step=1000.0)
    NumOfProducts = st.sidebar.slider("Number of Products", 1, 4, 1)
    HasCrCard = st.sidebar.selectbox("Has Credit Card?", [0, 1])
    IsActiveMember = st.sidebar.selectbox("Is Active Member?", [0, 1])
    EstimatedSalary = st.sidebar.number_input("Estimated Salary", min_value=0.0, max_value=250000.0, value=80000.0, step=1000.0)
    Complain = st.sidebar.selectbox("Filed Complaint?", [0, 1])
    SatisfactionScore = st.sidebar.slider("Satisfaction Score", 1, 5, 3)
    CardType = st.sidebar.selectbox("Card Type", ["SILVER", "GOLD", "PLATINUM", "DIAMOND"])
    PointEarned = st.sidebar.number_input("Reward Points", min_value=0, max_value=20000, value=3000, step=100)

    data = {
        "CreditScore": CreditScore,
        "Geography": Geography,
        "Gender": Gender,
        "Age": Age,
        "Tenure": Tenure,
        "Balance": Balance,
        "NumOfProducts": NumOfProducts,
        "HasCrCard": HasCrCard,
        "IsActiveMember": IsActiveMember,
        "EstimatedSalary": EstimatedSalary,
        "Complain": Complain,
        "Satisfaction Score": SatisfactionScore,
        "Card Type": CardType,
        "Point Earned": PointEarned
    }
    return pd.DataFrame(data, index=[0])

input_df = get_user_input()

st.subheader("Input Summary")
st.dataframe(input_df)

# ----- Prediction -----
if st.button("🔮 Predict Churn"):
    try:
        pred = model.predict(input_df)[0]
        proba = model.predict_proba(input_df)[0][1]

        st.markdown("---")
        st.subheader("Prediction Result")

        if pred == 1:
            st.error(f"⚠ This customer is **LIKELY TO CHURN**.\n\nEstimated probability: **{proba*100:.2f}%**")
        else:
            st.success(f"✅ This customer is **NOT LIKELY TO CHURN**.\n\nEstimated probability: **{proba*100:.2f}%**")

        # Simple text explanation
        st.markdown("---")
        st.markdown("### ⚙ Interpretation (high-level)")
        st.write(
            "- Higher complaints, lower satisfaction, and inactivity tend to increase churn risk.\n"
            "- More products, higher engagement, and good satisfaction usually reduce churn risk.\n"
            "- Use this prediction as a signal to **target retention offers** or follow-up with the customer."
        )

    except Exception as e:
        st.error(f"Error while predicting: {e}")
