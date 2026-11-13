import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

# 1. Charger le dataset original
df = pd.read_csv("dataset.csv")

print(df.head())
print("Nombre de lignes :", len(df))


# 2. Sélectionner aléatoirement 30 000 chansons
df_30k = df.sample(n=30000, random_state=42)

# Enregistrer le nouveau dataset
df_30k.to_csv("dataset_30000.csv", index=False)

print("Nombre de chansons sélectionnées :", len(df_30k))


# 3. Explorer le nouveau dataset
df = pd.read_csv("dataset_30000.csv")

print("\n--- Informations générales ---")
print(df.info())

print("\n--- Statistiques descriptives ---")
print(df.describe())

print("\n--- Valeurs manquantes ---")
print(df.isnull().sum())

print("\n--- Doublons ---")
print("Nombre de doublons :", df.duplicated().sum())


# 4. Vérifier les valeurs suspectes
print("\n--- Vérification des valeurs suspectes ---")

print("Durées égales à 0 :",
      (df["duration_ms"] == 0).sum())

print("Tempos égaux à 0 :",
      (df["tempo"] == 0).sum())

# Liste contenant uniquement les variables ayant des valeurs entre 0 et 1.
colonnes_01 = [
    "danceability",
    "energy",
    "speechiness",
    "acousticness",
    "instrumentalness",
    "liveness",
    "valence"
]

for colonne in colonnes_01:
    print(
        colonne,
        "hors intervalle [0, 1] :",
        ((df[colonne] < 0) | (df[colonne] > 1)).sum()
    )


# 5. Nettoyer les données

# Supprimer la colonne d'index inutile
df = df.drop(columns=["Unnamed: 0"])

# Supprimer les chansons ayant une durée ou un tempo égal à 0
df = df[
    (df["duration_ms"] > 0) &
    (df["tempo"] > 0)
]

# Vérifier le résultat
print("\n--- Après le nettoyage ---")
print("Nombre de chansons restantes :", len(df))

print("Durées égales à 0 :",
      (df["duration_ms"] == 0).sum())

print("Tempos égaux à 0 :",
      (df["tempo"] == 0).sum())

# Enregistrer le dataset nettoyé
df.to_csv("dataset_30000_clean.csv", index=False)

variables = [
    "danceability",
    "energy",
    "tempo",
    "loudness",
    "duration_ms",
    "track_genre",
    "popularity"
]

print(df[variables].isnull().sum())

print("Nombre final de chansons :", len(df))


# Créer l'histogramme
plt.figure(figsize=(8, 5))

plt.hist(df["popularity"], bins=20, edgecolor="black")

# Ajouter les titres et les étiquettes
plt.title("Distribution de la popularité des chansons")
plt.xlabel("Popularité (0 à 100)")
plt.ylabel("Nombre de chansons")

# Enregistrer le graphique
plt.savefig("distribution_popularite.png")

# Afficher le graphique
plt.show()


#Volet 2.3: Séparation des données

#Poser les variables
x = df[["danceability", "energy", "tempo", "loudness",
        "duration_ms", "track_genre"]]
y = df["popularity"]

#Séparer les données de test et d'entraînement
#Environ 80% des lignes vont à l'entraînemnt et 20% au test; random = 42 permet d'obtenir la même séparation à chaque fois qu'on relance le code
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 42)

#Volet 2.4: Établir une référence

#Calcule la popularité moyenne des données d'entraînement
moyenne = y_train.mean()

#Crée une prédiction pour chaque chanson de test; elles ont toutes la même valeur: la moyenne calculée juste avant
y_reference = [moyenne] * len(y_test)

#Calcul de la métrique d'évaluation MAE
#Compare les prédictions aux vraies popularités dans y_test
mae_reference = round(mean_absolute_error(y_test, y_reference), 2)

