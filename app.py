import streamlit as st
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pandas as pd
import pickle

# ------------------------ Load Model and Encoders ------------------------

model = tf.keras.models.load_model('model.h5')

with open('label_encoder_gender.pkl', 'rb') as file:
    label_encoder_gender = pickle.load(file)

with open('onehot_encoder_geo.pkl', 'rb') as file:
    onehot_encoder_geo = pickle.load(file)

with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

# ------------------------ Page Setup ------------------------

st.set_page_config(page_title="Customer Churn Predictor", layout="centered")
st.title('💼 Customer Churn Prediction using ANN')

st.markdown("""
This application predicts whether a bank customer is likely to **churn** (i.e., leave the bank), 
using an **Artificial Neural Network (ANN)** trained on customer data.
Fill in the customer's details below to get a prediction.
""")

# ------------------------ Input Fields with Help ------------------------

st.header("📋 Customer Information")

st.selectbox('🌍 Geography (Country of Residence)', 
             onehot_encoder_geo.categories_[0], 
             help="Country where the customer resides. Geography affects customer behavior.")

st.selectbox('🧑 Gender', 
             label_encoder_gender.classes_, 
             help="Biological sex of the customer.")

age = st.slider('🎂 Age', 18, 92, help="Customer’s age in years.")

credit_score = st.number_input('💳 Credit Score (300–850)', min_value=300, max_value=850, value=650,
                               help="A score representing creditworthiness. Higher is better.")

balance = st.number_input('💰 Account Balance', help="Customer’s current bank account balance.")

estimated_salary = st.number_input('📈 Estimated Annual Salary', help="Customer’s annual income.")

currency = st.selectbox("💱 Currency", ["EUR (€)", "USD ($)", "INR (₹)"],
                        help="Select the currency in which Balance and Salary are entered.")

tenure = st.slider('📆 Tenure with Bank (in years)', 0, 10, help="Number of years customer has stayed with the bank.")

num_of_products = st.slider('📦 Number of Products Used', 1, 4, 
                            help="Number of banking products the customer uses (e.g., savings, loans, credit card).")

has_cr_card = st.selectbox('💳 Has Credit Card?', [0, 1], 
                           help="Does the customer own a credit card? 1 = Yes, 0 = No.")

is_active_member = st.selectbox('📍 Is Active Member?', [0, 1], 
                                help="Is the customer actively using bank services? 1 = Yes, 0 = No.")

# ------------------------ Currency Conversion ------------------------

conversion_rates = {"EUR (€)": 1.0, "USD ($)": 0.93, "INR (₹)": 0.011}
conversion_rate = conversion_rates[currency]
balance_eur = balance * conversion_rate
salary_eur = estimated_salary * conversion_rate

st.caption(f"💶 Converted Balance: €{balance_eur:.2f} | Converted Salary: €{salary_eur:.2f}")

# ------------------------ Preprocessing ------------------------

input_data = pd.DataFrame({
    'CreditScore': [credit_score],
    'Gender': [label_encoder_gender.transform([st.session_state["🧑 Gender"]])[0]],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance_eur],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_cr_card],
    'IsActiveMember': [is_active_member],
    'EstimatedSalary': [salary_eur]
})

geo_encoded = onehot_encoder_geo.transform([[st.session_state["🌍 Geography (Country of Residence)"]]]).toarray()
geo_encoded_df = pd.DataFrame(geo_encoded, columns=onehot_encoder_geo.get_feature_names_out(['Geography']))
input_data = pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)

input_data_scaled = scaler.transform(input_data)

# ------------------------ Prediction ------------------------

prediction = model.predict(input_data_scaled)
prediction_proba = prediction[0][0]

# ------------------------ Output ------------------------

st.header("📊 Prediction Result")
st.write(f"**Churn Probability:** `{prediction_proba:.2f}`")

if prediction_proba > 0.5:
    st.error('⚠️ The customer is likely to churn.')
else:
    st.success('✅ The customer is likely to stay.')
