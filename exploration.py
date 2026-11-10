import pandas as pd

df = pd.read_csv("dataset_30000.csv")

print(df.head())
print("Nombre de lignes :", len(df))

# Examiner les types et les valeurs non nulles
df.info()

# Examiner les statistiques descriptives
print(df.describe())
print(df.isnull().sum())