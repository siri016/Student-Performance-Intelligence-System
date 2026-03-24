import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("data/Student_Performance_Dataset.csv")

print("First 5 rows:\n", df.head())
print("\nDataset Info:\n")
print(df.info())

print("\nMissing Values:\n", df.isnull().sum())

df.fillna(df.mean(numeric_only=True), inplace=True)

print("\nStatistical Summary:\n", df.describe())

for col in df.select_dtypes(include=np.number).columns:
    plt.figure()
    sns.histplot(df[col], kde=True)
    plt.title(f"Distribution of {col}")
    plt.savefig(f"notebooks/{col}_distribution.png")

corr = df.corr(numeric_only=True)

print("\nCorrelation Matrix:\n", corr)

plt.figure(figsize=(10,8))
sns.heatmap(corr, annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.savefig("notebooks/correlation_heatmap.png")

target = "Performance" 

if target in corr.columns:
    important_features = corr[target].sort_values(ascending=False)
    print("\nImportant Features:\n", important_features)
else:
    print("\n⚠️ Update target column name in code!")
