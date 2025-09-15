from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
import xgboost as xgb
import webbrowser
import threading

# Initialize the app
app = FastAPI()

# Load the model
model = xgb.XGBClassifier()
model.load_model("trained_diabetes_risk_model.json")

# Define the input data schema
class ModelInput(BaseModel):
    high_blood_pressure: int
    general_health: int
    high_cholesterol: int
    age: int
    cholesterol_check: int
    heavy_alcohol_consumption: int
    heart_disease_or_attack: int
    body_mass_index: int
    difficulty_walking: int
    sex: int
    income: int
    stroke: int
    education: int
    physical_health: int
    mental_health: int
    smoker: int
    no_doctor_because_of_cost: int
    any_healthcare: int

@app.get("/")
def read_root():
    return {"message": "Model is ready!"}

@app.post("/predict")
def predict(input_data: ModelInput):
    input_array = np.array([[
        input_data.high_blood_pressure,
        input_data.general_health,
        input_data.high_cholesterol,
        input_data.age,
        input_data.cholesterol_check,
        input_data.heavy_alcohol_consumption,
        input_data.heart_disease_or_attack,
        input_data.body_mass_index,
        input_data.difficulty_walking,
        input_data.sex,
        input_data.income,
        input_data.stroke,
        input_data.education,
        input_data.physical_health,
        input_data.mental_health,
        input_data.smoker,
        input_data.no_doctor_because_of_cost,
        input_data.any_healthcare
    ]])

    # Get probability of diabetes
    proba = model.predict_proba(input_array)[0][1]
    risk_score = float(proba)

    # Map probability to 5-class risk
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

    predicted_class = int(model.predict(input_array)[0])

    return {
        "predicted_class": predicted_class,
        "risk_score": risk_score * 100,
        "likelihood": likelihood
    }

def open_browser():
    webbrowser.open_new("http://127.0.0.1:8000/docs")

if __name__ == "__main__":
    threading.Timer(1.5, open_browser).start()
    import uvicorn
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)