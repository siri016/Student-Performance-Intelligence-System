
import pandas as pd
from sklearn.preprocessing import StandardScaler


# 1. Load dataset

df = pd.read_csv("Student_Performance_Dataset.csv")


# 2. Store Student_ID separately

student_ids = df["Student_ID"]


df = df.drop("Student_ID", axis=1)


# 3. Check basic info

print("Shape:", df.shape)
print("Missing values:\n", df.isnull().sum())


# 4. Remove duplicates (if any)

df = df.drop_duplicates()


# 5. Encode categorical columns

categorical_cols = df.select_dtypes(include=['object']).columns

df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)


# 6. Feature Scaling

scaler = StandardScaler()
df_scaled = scaler.fit_transform(df)

df = pd.DataFrame(df_scaled, columns=df.columns)


# 7. Add Student_ID back (for tracking)

df["Student_ID"] = student_ids.values


# 8. Save cleaned dataset

df.to_csv("data/cleaned_student_data.csv", index=False)

print("✅ Preprocessing completed successfully!")