from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import joblib

app = FastAPI(title="Diabetes Prediction API")

# Load the model
model = joblib.load("diabetes_model.joblib")

# Define expected input format
class PatientData(BaseModel):
    BMI: float
    TEZINA: float
    VISINA: float
    PRITISAK_DIASTOLIC: float
    PRITISAK_SISTOLIC: float
    Glukoza: float
    HbA1c: float
    Insulin: float
    TSH: float
    FT3: float
    FT4: float
    Trigliceridi: float
    HDL: float
    LDL: float
    Holesterol: float
    GENDER: int
    PUSENJE: int

@app.post("/predict")
def predict(data: PatientData):
    input_data = np.array([[getattr(data, field) for field in data.__annotations__]])
    prediction = model.predict(input_data)[0]
    return {"prediction": "Diabetes: Yes" if prediction == 1 else "Diabetes: No"}

    
    prediction = model.predict(input_data)[0]

    result = "Diabetes: Yes" if prediction == 1 else "Diabetes: No"
    return {"prediction": result}
