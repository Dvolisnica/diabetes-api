from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

# Učitavanje modela
model = joblib.load("diabetes_model.joblib")

# Inicijalizacija FastAPI aplikacije
app = FastAPI(title="Diabetes Prediction API")

# Definisanje strukture inputa
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
    # Pretvaranje inputa u format koji model očekuje
    input_data = np.array([[data.BMI, data.TEZINA, data.VISINA, data.PRITISAK_DIASTOLIC, 
                             data.PRITISAK_SISTOLIC, data.Glukoza, data.HbA1c, data.Insulin,
                             data.TSH, data.FT3, data.FT4, data.Trigliceridi, data.HDL, 
                             data.LDL, data.Holesterol, data.GENDER, data.PUSENJE]])
    
    prediction = model.predict(input_data)[0]

    result = "Diabetes: Yes" if prediction == 1 else "Diabetes: No"
    return {"prediction": result}
