from fastapi import FastAPI
import pandas as pd
import pickle

app = FastAPI()

# Load model and scaler
model = pickle.load(open("../dnn_model/model.pkl", "rb"))
scaler = pickle.load(open("../dnn_model/scaler.pkl", "rb"))

@app.get("/")
def home():
    return {"message": "Student Performance Prediction API"}

@app.post("/predict")
def predict(data: dict):
    try:
        # Convert input to DataFrame
        df = pd.DataFrame([data])

        # Convert categorical
        df = pd.get_dummies(df)

        # Scale
        df_scaled = scaler.transform(df)

        # Predict
        prediction = model.predict(df_scaled)

        return {
            "Predicted Final Percentage": float(prediction[0])
        }

    except Exception as e:
        return {"error": str(e)}
