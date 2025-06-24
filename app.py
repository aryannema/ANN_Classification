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
""")

# Sidebar with context
st.sidebar.title("ℹ️ About This Project")
st.sidebar.markdown("""
### 🔍 What is Churn?
**Customer churn** refers to when a customer decides to stop using a company’s services — in this case, leave the bank.

### 🧠 Model Used
This app uses an **Artificial Neural Network (ANN)** trained on customer data to predict churn.

---

### 📊 Input Feature Guide

Here’s what each input means:

- **Geography** 🌍  
  Country where the customer resides (e.g., France, Germany, Spain)

- **Gender** 🧑  
  Biological sex of the customer — helps understand demographics

- **Age** 🎂  
  Customer's age (between 18 and 92 in our data)

- **Credit Score** 💳  
  A number (300–850) that represents how reliable a person is at repaying borrowed money.  
  *Higher score = more trustworthy.*

- **Balance** 💰  
  The amount of money the customer currently holds in their bank account.  
  *Values are in Euros (€).*

- **Estimated Salary** 📈  
  The customer’s yearly income (in Euros).  
  Helps estimate financial standing and likelihood to churn.

- **Tenure** 📆  
  How many years the customer has been with the bank.

- **Number of Products** 📦  
  Number of bank products used (loans, credit cards, etc.)

- **Has Credit Card** 💳  
  `1 = Yes`, `0 = No` — whether the customer has a credit card

- **Is Active Member** 📍  
  `1 = Yes`, `0 = No` — indicates if the customer regularly engages with the bank’s services.

---

### 💡 How to Use

1. Choose the currency and enter customer details.
2. The app will show:
   - A **churn probability**
   - A decision: likely to churn or not

📝 *Note: For educational purposes only.*
""")

# ------------------------ Input Fields ------------------------

st.header("📋 Enter Customer Information")

geography = st.selectbox('🌍 Geography', onehot_encoder_geo.categories_[0])
gender = st.selectbox('🧑 Gender', label_encoder_gender.classes_)
age = st.slider('🎂 Age (years)', 18, 92)
credit_score = st.number_input('💳 Credit Score (300–850)', min_value=300, max_value=850, value=650)
balance = st.number_input('💰 Current Balance')
estimated_salary = st.number_input('📈 Estimated Annual Salary')
tenure = st.slider('📆 Tenure (Years with Bank)', 0, 10)
num_of_products = st.slider('📦 Number of Products Used', 1, 4)
has_cr_card = st.selectbox('💳 Has Credit Card?', [0, 1])
is_active_member = st.selectbox('📍 Is Active Member?', [0, 1])

# ------------------------ Currency Conversion ------------------------

currency = st.selectbox("💱 Select Currency", ["EUR (€)", "USD ($)", "INR (₹)"])

# Approximate conversion rates to EUR
conversion_rates = {
    "EUR (€)": 1.0,
    "USD ($)": 0.93,
    "INR (₹)": 0.011
}

conversion_rate = conversion_rates[currency]
balance_eur = balance * conversion_rate
salary_eur = estimated_salary * conversion_rate

st.info(f"🔁 Converted Balance = €{balance_eur:.2f}, Salary = €{salary_eur:.2f}")

# ------------------------ Data Preprocessing ------------------------

input_data = pd.DataFrame({
    'CreditScore': [credit_score],
    'Gender': [label_encoder_gender.transform([gender])[0]],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance_eur],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_cr_card],
    'IsActiveMember': [is_active_member],
    'EstimatedSalary': [salary_eur]
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
