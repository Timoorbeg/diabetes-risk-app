# streamlit_app.py
import streamlit as st
import xgboost as xgb
import numpy as np

# Load your trained model
model = xgb.XGBClassifier()
model.load_model("trained_diabetes_risk_model.json")

st.title("Diabetes Risk Predictor")

# Input fields for all features
high_blood_pressure = st.number_input("High Blood Pressure (0/1)", min_value=0, max_value=1)
general_health = st.number_input("General Health (1-5)", min_value=1, max_value=5)
high_cholesterol = st.number_input("High Cholesterol (0/1)", min_value=0, max_value=1)
age = st.number_input("Age", min_value=0, max_value=120)
cholesterol_check = st.number_input("Cholesterol Check (0/1)", min_value=0, max_value=1)
heavy_alcohol_consumption = st.number_input("Heavy Alcohol Consumption (0/1)", min_value=0, max_value=1)
heart_disease_or_attack = st.number_input("Heart Disease/Attack (0/1)", min_value=0, max_value=1)
body_mass_index = st.number_input("BMI", min_value=0, max_value=250)
difficulty_walking = st.number_input("Difficulty Walking (0/1)", min_value=0, max_value=1)
sex = st.number_input("Sex (0=Female, 1=Male)", min_value=0, max_value=1)
income = st.number_input("Income level (1-8)", min_value=1, max_value=8)
stroke = st.number_input("Stroke history (0/1)", min_value=0, max_value=1)
education = st.number_input("Education level (1-6)", min_value=1, max_value=6)
physical_health = st.number_input("Physical health (0-30 days)", min_value=0, max_value=30)
mental_health = st.number_input("Mental health (0-30 days)", min_value=0, max_value=30)
smoker = st.number_input("Smoker (0/1)", min_value=0, max_value=1)
no_doctor_because_of_cost = st.number_input("No doctor due to cost (0/1)", min_value=0, max_value=1)
any_healthcare = st.number_input("Access to healthcare (0/1)", min_value=0, max_value=1)

# Predict button
if st.button("Predict Risk"):
    input_array = np.array([[high_blood_pressure, general_health, high_cholesterol, age,
                             cholesterol_check, heavy_alcohol_consumption, heart_disease_or_attack,
                             body_mass_index, difficulty_walking, sex, income, stroke,
                             education, physical_health, mental_health, smoker,
                             no_doctor_because_of_cost, any_healthcare]])
    
    # Prediction
    risk_score = float(model.predict_proba(input_array)[0][1])
    predicted_class = int(model.predict(input_array)[0])
    
    # Likelihood mapping
    if risk_score < 0.2:
        likelihood = "Very Low"
    elif risk_score < 0.4:
        likelihood = "Low"
    elif risk_score < 0.6:
        likelihood = "Moderate"
    elif risk_score < 0.8:
        likelihood = "High"
    else:
        likelihood = "Very High"

    # Display results
    st.success(f"Predicted Class: {predicted_class}")
    st.info(f"Risk Score: {risk_score*100:.2f}%")
    st.warning(f"Likelihood: {likelihood}")