import streamlit as st
import numpy as np
import pandas as pd
import pickle
import os
from tensorflow.keras.models import load_model

# Page Configuration
st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="📊",
    layout="centered",
    initial_sidebar_state="expanded"
)

st.title("📊 Customer Churn Prediction System")
st.write("Artificial Neural Network (ANN) aur Sigmoid Activation Function dwara customer churn ki sambhavna predict karein.")
st.markdown("---")

# Asset Files Loading Function
@st.cache_resource
def load_ann_assets():
    model_path = 'churn_ann_model.keras'
    scaler_path = 'scaler.pkl'
    
    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        return None, None
    
    model = load_model(model_path)
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    return model, scaler

model, scaler = load_ann_assets()

if model is None or scaler is None:
    st.error("❌ Model ya Scaler file nahi mili! Kripya pehle `train.py` run karke `churn_ann_model.keras` aur `scaler.pkl` generate karein.")
else:
    st.subheader("📝 Customer Details Enter Karein")
    
    # Form Layout using Columns
    col1, col2 = st.columns(2)
    
    with col1:
        credit_score = st.number_input("Credit Score", min_value=300, max_value=850, value=650, step=1)
        age = st.number_input("Age (Years)", min_value=18, max_value=100, value=35, step=1)
        tenure = st.number_input("Tenure (Years)", min_value=0, max_value=10, value=5, step=1)
        balance = st.number_input("Account Balance (₹)", min_value=0.0, max_value=200000.0, value=50000.0, step=1000.0)

    with col2:
        num_of_products = st.selectbox("Number of Products", options=[1, 2, 3, 4], index=0)
        has_cr_card = st.selectbox("Has Credit Card?", options=["Yes", "No"], index=0)
        is_active_member = st.selectbox("Is Active Member?", options=["Yes", "No"], index=0)
        estimated_salary = st.number_input("Estimated Salary (₹)", min_value=0.0, max_value=200000.0, value=60000.0, step=1000.0)

    # Encode categorical dropdowns
    has_cr_card_val = 1 if has_cr_card == "Yes" else 0
    is_active_member_val = 1 if is_active_member == "Yes" else 0

    st.markdown("---")
    
    if st.button("🔍 Predict Churn Risk", use_container_width=True):
        # 1. Input Features DataFrame in Exact Training Order
        input_data = pd.DataFrame([{
            'CreditScore': credit_score,
            'Age': age,
            'Tenure': tenure,
            'Balance': balance,
            'NumOfProducts': num_of_products,
            'HasCrCard': has_cr_card_val,
            'IsActiveMember': is_active_member_val,
            'EstimatedSalary': estimated_salary
        }])
        
        # 2. Apply Standard Scaling
        scaled_features = scaler.transform(input_data)
        
        # 3. Predict via ANN Model (Sigmoid Probability Output)
        prediction_prob = float(model.predict(scaled_features)[0][0])
        
        # 4. Results Display
        st.subheader("🎯 Prediction Result")
        
        st.write(f"**Churn Probability:** `{prediction_prob * 100:.2f}%`")
        st.progress(prediction_prob)
        
        # Classification Threshold (>= 0.5 is Churn)
        if prediction_prob >= 0.5:
            st.error(f"⚠️ **High Churn Risk!** Customer ke company chhodne ki sambhavna zyada hai.")
        else:
            st.success(f"✅ **Low Churn Risk!** Customer company ke sath bana rahega.")