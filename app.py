import streamlit as st
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pandas as pd
import pickle

# ------------------------ Load Trained Models and Scalers ------------------------

# Load the trained Keras ANN model
model = tf.keras.models.load_model('model.h5')

# Load the encoders and scaler
with open('label_encoder_gender.pkl', 'rb') as file:
    label_encoder_gender = pickle.load(file)

with open('onehot_encoder_geo.pkl', 'rb') as file:
    onehot_encoder_geo = pickle.load(file)

with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

# ------------------------ Streamlit App Interface ------------------------

# App title and basic description
st.set_page_config(page_title="Churn Predictor", layout="centered")
st.title('💼 Customer Churn Prediction')
st.write("""
This application predicts whether a customer is likely to leave the bank or stay, 
based on inputs like Credit Score, Geography, Age, Balance, and more.
""")

# Sidebar explanation for context
st.sidebar.title("ℹ️ About This Project")
st.sidebar.markdown("""
This is an Artificial Neural Network (ANN)-based model trained on customer data to 
predict churn (whether a customer will leave the bank).

The model uses features such as:
- Geography
- Gender
- Age
- Balance
- Credit Score
- Salary
- Products used
- Activity & Card status

**Note:** This is a demo app and predictions should not be used for real financial decisions.
""")

# ------------------------ User Input Section ------------------------

st.header("📋 Enter Customer Details")

geography = st.selectbox('Geography', onehot_encoder_geo.categories_[0])
gender = st.selectbox('Gender', label_encoder_gender.classes_)
age = st.slider('Age', 18, 92)
balance = st.number_input('Balance')
credit_score = st.number_input('Credit Score')
estimated_salary = st.number_input('Estimated Salary')
tenure = st.slider('Tenure (Years with Bank)', 0, 10)
num_of_products = st.slider('Number of Products', 1, 4)
has_cr_card = st.selectbox('Has Credit Card?', [0, 1])
is_active_member = st.selectbox('Is Active Member?', [0, 1])

# ------------------------ Data Preparation ------------------------

# Construct DataFrame for model input
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

# One-hot encode 'Geography'
geo_encoded = onehot_encoder_geo.transform([[geography]]).toarray()
geo_encoded_df = pd.DataFrame(geo_encoded, columns=onehot_encoder_geo.get_feature_names_out(['Geography']))

# Combine all features
input_data = pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)

# Scale the features (standardization)
input_data_scaled = scaler.transform(input_data)

# ------------------------ Make Prediction ------------------------

prediction = model.predict(input_data_scaled)
prediction_proba = prediction[0][0]

# ------------------------ Show Output ------------------------

st.header("🔍 Prediction Result")

st.write(f"**Churn Probability:** `{prediction_proba:.2f}`")

if prediction_proba > 0.5:
    st.error('⚠️ The customer is likely to churn.')
else:
    st.success('✅ The customer is likely to stay.')

