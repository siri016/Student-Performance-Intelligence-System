import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, VotingRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# -----------------------------
# 1. Load cleaned dataset
# -----------------------------
df = pd.read_csv("data/cleaned_student_data.csv")

# -----------------------------
# 2. Define features and target
# -----------------------------
X = df.drop([
    "Final_Percentage",
    "Student_ID",
    "Math_Score",
    "Science_Score",
    "English_Score"
], axis=1)

y = df["Final_Percentage"]

# -----------------------------
# 3. Train-test split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -----------------------------
# 4. Linear Regression
# -----------------------------
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

lr_pred = lr_model.predict(X_test)

print("🔹 Linear Regression")
print("R2 Score:", r2_score(y_test, lr_pred))
print("RMSE:", np.sqrt(mean_squared_error(y_test, lr_pred)))
print("MAE:", mean_absolute_error(y_test, lr_pred))

# -----------------------------
# 5. Random Forest
# -----------------------------
rf_model = RandomForestRegressor(
    n_estimators=200,
    max_depth=10,
    random_state=42
)

rf_model.fit(X_train, y_train)

rf_pred = rf_model.predict(X_test)

print("\n🔹 Random Forest")
print("R2 Score:", r2_score(y_test, rf_pred))
print("RMSE:", np.sqrt(mean_squared_error(y_test, rf_pred)))
print("MAE:", mean_absolute_error(y_test, rf_pred))

# -----------------------------
# 6. Ensemble Model (Voting)
# -----------------------------
ensemble_model = VotingRegressor(
    estimators=[
        ('lr', lr_model),
        ('rf', rf_model)
    ]
)

ensemble_model.fit(X_train, y_train)

ensemble_pred = ensemble_model.predict(X_test)

print("\n🔥 Ensemble Model (Linear + RF)")
print("R2 Score:", r2_score(y_test, ensemble_pred))
print("RMSE:", np.sqrt(mean_squared_error(y_test, ensemble_pred)))
print("MAE:", mean_absolute_error(y_test, ensemble_pred))