import pandas as pd

df = pd.read_csv("dataset.csv")

print(df.head())
print("Nombre de lignes :", len(df))


# Sélectionner aléatoirement 30 000 chansons
df_30k = df.sample(n=30000, random_state=42)

# Enregistrer le nouveau dataset
df_30k.to_csv("dataset_30000.csv", index=False)


