# Simple DNN using sklearn (MLPRegressor)

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.inspection import permutation_importance

# -------- Load Dataset --------
df = pd.read_csv('../data/cleaned_student_data.csv')

print("Columns:", df.columns)

# -------- Data Preprocessing --------
df = df.dropna()

# Remove ID Column
df = df.drop('Student_ID', axis=1)

# 🔥 CHANGE THIS AFTER SEEING COLUMNS
target_column = 'Final_Percentage'   # change if needed

X = df.drop(target_column, axis=1)
y = df[target_column]

# Convert categorical to numeric
X = pd.get_dummies(X)

# Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train test split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# -------- DNN Model (MLP) --------
model = MLPRegressor(
    hidden_layer_sizes=(64, 32),
    activation='relu',
    max_iter=500,
    random_state=42
)

model.fit(X_train, y_train)

# -------- Evaluation --------
score = model.score(X_test, y_test)
print("\nModel Score:", score)

# -------- Prediction --------
predictions = model.predict(X_test[:5])
print("\nSample Predictions:\n", predictions)

# -------- Feature Importance --------
r = permutation_importance(
    model,
    X_test,
    y_test,
    n_repeats=5,
    random_state=42
)

importance_df = pd.DataFrame({
    'Feature': pd.get_dummies(df.drop(target_column, axis=1)).columns,
    'Importance': r.importances_mean
}).sort_values(by='Importance', ascending=False)

print("\nTop Influencing Factors:")
print(importance_df.head(10))



import pickle

pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(scaler, open("scaler.pkl", "wb"))

print("Model saved!")
