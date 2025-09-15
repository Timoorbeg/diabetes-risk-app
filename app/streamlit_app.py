import streamlit as st
import requests

st.title("Diabetes Risk Prediction")

# Input fields for all features
high_blood_pressure = st.number_input("High Blood Pressure (0/1)", min_value=0, max_value=1)
general_health = st.number_input("General Health (1-5)", min_value=1, max_value=5)
high_cholesterol = st.number_input("High Cholesterol (0/1)", min_value=0, max_value=1)
age = st.number_input("Age", min_value=0, max_value=13)
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

if st.button("Predict Risk"):
    payload = {
        "high_blood_pressure": high_blood_pressure,
        "general_health": general_health,
        "high_cholesterol": high_cholesterol,
        "age": age,
        "cholesterol_check": cholesterol_check,
        "heavy_alcohol_consumption": heavy_alcohol_consumption,
        "heart_disease_or_attack": heart_disease_or_attack,
        "body_mass_index": body_mass_index,
        "difficulty_walking": difficulty_walking,
        "sex": sex,
        "income": income,
        "stroke": stroke,
        "education": education,
        "physical_health": physical_health,
        "mental_health": mental_health,
        "smoker": smoker,
        "no_doctor_because_of_cost": no_doctor_because_of_cost,
        "any_healthcare": any_healthcare
    }
    
    try:
        response = requests.post("http://127.0.0.1:8000/predict", json=payload)
        if response.status_code == 200:
            result = response.json()
            st.success(f"Predicted Class: {result['predicted_class']}")
            st.info(f"Risk Score: {result['risk_score']:.2f}%")
            st.warning(f"Likelihood: {result['likelihood']}")
        else:
            st.error(f"API returned an error: {response.status_code} - {response.text}")
    except Exception as e:
        st.error(f"Error calling API: {e}")