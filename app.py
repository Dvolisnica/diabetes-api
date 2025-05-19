from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import joblib

app = FastAPI(title="Diabetes Prediction API")

# Load the model
model = joblib.load("diabetes_model.joblib")

# Define expected input format

class PatientData(BaseModel):
    NHC:float
    GENDER: int
    BMI: float
    Glukoza: float
    HbA1c: float
    Trigliceridi: float
    HDL: float
    SHBG: float
    ALT: float
    AST: float

@app.post("/predict")
def predict(data: PatientData):
    input_array = np.array([[data.NHC,data.GENDER, data.BMI, data.Glukoza, data.HbA1c,
                             data.Trigliceridi, data.HDL, data.SHBG, data.ALT, data.AST]])
    
    probability = model.predict_proba(input_array)[0][1]  # vjerovatnoća za klasu 1
    percentage = round(probability * 100, 2)

    return {
        "prediction": "Diabetes: Yes" if percentage >= 50 else "Diabetes: No",
        "probability": f"{percentage} %"
    }

