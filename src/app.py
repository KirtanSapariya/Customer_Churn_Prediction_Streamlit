import streamlit as st
import joblib
import numpy as np

scaler = joblib.load("scaler.pkl")
model = joblib.load("model.pkl")

st.title("Churn Prediction App")

st.divider()

st.write("Enter the customer details below:")

st.divider()

age = st.number_input("Age", min_value=10, max_value=100, value=30)

tenure = st.number_input("Tenure", min_value=0, max_value=100, value=12)

monthly_charges = st.number_input("Monthly Charges", min_value=0.0, max_value=100.00, value=50.0)

gender = st.selectbox("Gender", ["Male", "Female"])

st.divider()

prediction_button = st.button("Predict Churn")

st.divider()

if prediction_button:
    gender_selected = 1 if gender == "Female" else 0

    X = [age, gender_selected, tenure, monthly_charges]
    X1 = np.array(X)
    X_array = scaler.transform([X1])
    prediction = model.predict(X_array)[0]
    predicted = "Yes" if prediction == 1 else "No"
    st.balloons()
    st.write(f"Churn Prediction: {predicted}")

else:
    st.write("Please click the 'Predict Churn' button to see the prediction.")