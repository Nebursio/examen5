import pickle
from fastapi import FastAPI
from pydantic import BaseModel

# Cargar el modelo SVM guardado
with open('svm_model.pkl', 'rb') as f:
    svm_model = pickle.load(f)
class InputData(BaseModel):
    Pregnancies: float
    Glucose: float
    BloodPressure: float
    SkinThickness: float
    Insulin: float
    BMI: float
    DiabetesPedigreeFunction: float
    Age: float
# Crear una instancia de la aplicación FastAPI
app = FastAPI()

# Definir el punto final de la API para realizar predicciones
@app.post("/predict")
def predict(data: InputData):

    input_features = [data.Pregnancies, data.Glucose, data.BloodPressure, data.SkinThickness,
                      data.Insulin, data.BMI, data.DiabetesPedigreeFunction, data.Age]
    input_array = [input_features]


    prediction = svm_model.predict(input_array)

    # Devolver la predicción como resultado de la API
    return {"prediction": int(prediction[0])}


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000)
