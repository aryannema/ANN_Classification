# 💼 Customer Churn Prediction using ANN

This is an interactive web application built using **Streamlit** that predicts whether a bank customer is likely to **churn** (i.e., leave the bank) based on various personal and financial attributes. The backend uses a trained **Artificial Neural Network (ANN)** model built with **TensorFlow**.

---

## 🔍 Overview

Customer churn is a major concern in the banking sector. Predicting which customers are likely to leave allows the bank to take proactive measures to retain them. This project demonstrates how machine learning and deep learning can be used to solve such problems.

The app takes in customer details and returns:

- A **churn probability**
- A decision: "Likely to Churn" or "Likely to Stay"

---

## 🧠 Model

- Framework: `TensorFlow` / `Keras`
- Architecture: Artificial Neural Network (ANN)
- Trained on: Structured customer data
- Input preprocessing: OneHotEncoding + LabelEncoding + StandardScaler

---

## 📊 Features Used

| Feature           | Description                                                     |
| ----------------- | --------------------------------------------------------------- |
| `Geography`       | Country where the customer resides (France, Germany, Spain)     |
| `Gender`          | Biological sex of the customer                                  |
| `Age`             | Customer's age (18–92)                                          |
| `CreditScore`     | A score representing creditworthiness (300–850)                 |
| `Balance`         | Customer’s bank account balance (converted to €)                |
| `EstimatedSalary` | Annual salary (converted to €)                                  |
| `Tenure`          | Years the customer has been with the bank                       |
| `NumOfProducts`   | Number of bank products the customer uses                       |
| `HasCrCard`       | 1 = Yes, 0 = No                                                 |
| `IsActiveMember`  | 1 = Yes, 0 = No (indicates if they regularly use bank services) |

> 💡 **Note:** The app allows users to enter `Balance` and `Salary` in USD, INR, or EUR and automatically converts them to Euros for the model.

---

## 🚀 How to Run the App Locally

1. **Clone the repository**

   ```bash
   git clone https://github.com/yourusername/ANN_Classification
   cd ANN_Classification
   ```

2. **Install dependencies**
   It's recommended to use a virtual environment.

   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Streamlit app**
   ```bash
   streamlit run app.py
   ```

---

## 📦 Requirements

The `requirements.txt` file includes:

```
streamlit
tensorflow==2.15.0
scikit-learn
pandas
numpy
```

---

## 📁 Project Structure

```
├── app.py
├── model.h5
├── scaler.pkl
├── label_encoder_gender.pkl
├── onehot_encoder_geo.pkl
├── requirements.txt
└── README.md
```

---

## 📝 License

This project is for **educational purposes only** and should not be used for real financial decision-making without validation.

---

## 🙋‍♂️ Author

Made by [Aryan Nema](https://github.com/aryannema)
