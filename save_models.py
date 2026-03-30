import sys
sys.path.append('src')
from model_training import lr_model, rf_model, ensemble_model
import joblib
import os

os.makedirs("models", exist_ok=True)
joblib.dump(lr_model, "models/lr_model.pkl")
joblib.dump(rf_model, "models/rf_model.pkl")
joblib.dump(ensemble_model, "models/ensemble_model.pkl")
print("✅ Models saved successfully!")
