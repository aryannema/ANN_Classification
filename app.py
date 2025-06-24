import streamlit as st
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pandas as pd
import pickle

# ------------------------ Load Trained Model and Encoders ------------------------

model = tf.keras.models.load_model('model.h5')

with open('label_encoder_gender.pkl', 'rb') as file:
    label_encoder_gender = pickle.load(file)

with open('onehot_encoder_geo.pkl', 'rb') as file:
    onehot_encoder_geo = pickle.load(file)

with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

# ------------------------ Streamlit Setup ------------------------

st.set_page_config(page_title="Customer Churn Predictor", layout="centered")
st.title('💼 Customer Churn Prediction Using ANN')
st.write("""
This application uses an **Artificial Neural Network (ANN)** to predict whether a bank customer 
is likely to **churn** (i.e., leave the bank). It is trained on real customer data 
with features like Age, Geography, Salary, Credit Score, etc.

To get started, enter the customer's information in the fields below.
""")

# Sidebar with context
st.sidebar.title("ℹ️ About This Project")
st.sidebar.markdown("""
**🔍 What is Churn?**
> Churn is when a customer decides to leave a service or company — in this case, a bank.

**🧠 Model Used**
> This is an Artificial Neural Network built using TensorFlow/Keras.

**📊 Inputs Explained:**
- `Geography`: Country of the customer (used for behavior pattern)
- `Gender`: Male or Female
- `Age`: Customer's age
- `Balance`: Amount in the customer's account
- `Credit Score`: A score representing creditworthiness
- `Estimated Salary`: Customer’s annual salary
- `Tenure`: How many years the customer has been with the bank
- `Number of Products`: Number of bank products used (loans, savings, credit card, etc.)
- `Has Credit Card`: Whether the customer has a credit card (`1`: Yes, `0`: No)
- `Is Active Member`: Is the customer actively using bank services?

**📌 How to Use:**
Fill in the fields below to get a churn prediction. The app will return the probability 
of churn and a decision (Likely to churn or not).
""")

# ------------------------ Input Fields ------------------------

st.header("📋 Enter Customer Information")

geography = st.selectbox('🌍 Geography', onehot_encoder_geo.categories_[0])
gender = st.selectbox('🧑 Gender', label_encoder_gender.classes_)
age = st.slider('🎂 Age (years)', 18, 92)
credit_score = st.number_input('💳 Credit Score (300-850 recommended)', min_value=300, max_value=850, value=650)
balance = st.number_input('💰 Current Balance in Account')
estimated_salary = st.number_input('📈 Estimated Annual Salary')
tenure = st.slider('📆 Tenure (Years with Bank)', 0, 10)
num_of_products = st.slider('📦 Number of Products Used', 1, 4)
has_cr_card = st.selectbox('💳 Has Credit Card?', [0, 1])
is_active_member = st.selectbox('📍 Is Active Member?', [0, 1])

# ------------------------ Data Preprocessing ------------------------

input_data = pd.DataFrame({
    'CreditScore': [credit_score],
    'Gender': [label_encoder_gender.transform([gender])[0]],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_cr_card],
    'IsActiveMember': [is_active_member],
    'EstimatedSalary': [estimated_salary]
})

geo_encoded = onehot_encoder_geo.transform([[geography]]).toarray()
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

