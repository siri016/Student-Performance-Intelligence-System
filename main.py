from fastapi import FastAPI
import joblib
import pandas as pd
from pydantic import BaseModel

app = FastAPI(title="Student Performance Intelligence System")

# Load models
rf_model = joblib.load("models/rf_model.pkl")
ensemble_model = joblib.load("models/ensemble_model.pkl")

# Input schema
class StudentInput(BaseModel):
    Age: float
    Class: float
    Study_Hours_Per_Day: float
    Attendance_Percentage: float
    Previous_Year_Score: float
    Gender_Male: float
    Parental_Education_High_School: float
    Parental_Education_Postgraduate: float
    Internet_Access_Yes: float
    Extracurricular_Activities_Yes: float
    Performance_Level_Excellent: float
    Performance_Level_Good: float
    Performance_Level_Poor: float
    Pass_Fail_Pass: float

# Home route
@app.get("/")
def home():
    return {"message": "Student Performance API is running!"}

# Recommendation function
def generate_recommendations(input_df):
    rec = []

    if input_df["Study_Hours_Per_Day"][0] < 3:
        rec.append("Increase study hours")

    if input_df["Attendance_Percentage"][0] < 75:
        rec.append("Improve attendance")

    return rec

# Prediction API
@app.post("/predict")
def predict(student: StudentInput):
    try:
        input_df = pd.DataFrame([student.dict()])

        # Fix column name issue
        input_df.rename(columns={
            "Parental_Education_High_School": "Parental_Education_High School"
        }, inplace=True)

        # 🔥 Prediction (convert to percentage)
        predicted_score = float(ensemble_model.predict(input_df)[0]) * 100

        # 🔥 Ensure value between 0–100
        if predicted_score < 0:
            predicted_score = 0
        elif predicted_score > 100:
            predicted_score = 100

        # Performance category
        if predicted_score >= 80:
            category = "Excellent"
        elif predicted_score >= 60:
            category = "Good"
        else:
            category = "Needs Improvement"

        # Recommendations
        recommendations = generate_recommendations(input_df)

        # Feature importance
        features = rf_model.feature_names_in_.tolist()
        importance = rf_model.feature_importances_
        top_factors = sorted(zip(features, importance),
                             key=lambda x: x[1], reverse=True)[:5]

        return {
            "predicted_score": round(predicted_score, 2),
            "performance_category": category,
            "recommendations": recommendations,
            "top_influencing_factors": [
                {"factor": f, "importance_percent": round(i * 100, 2)}
                for f, i in top_factors
            ]
        }

    except Exception as e:
        return {"error": str(e)}

# Feature importance API
@app.get("/feature-importance")
def feature_importance():
    features = rf_model.feature_names_in_.tolist()
    importance = rf_model.feature_importances_
    ranked = sorted(zip(features, importance),
                    key=lambda x: x[1], reverse=True)

    return {
        "key_factors": [
            {"factor": f, "importance_percent": round(i * 100, 2)}
            for f, i in ranked
        ]
    }