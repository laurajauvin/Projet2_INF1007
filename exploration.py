import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestRegressor
import json
import joblib
import os




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
mae_reference = mean_absolute_error(y_test, y_reference)


#Volet 2.5: Entraîner et évaluer

#Transformation de track_genre en valeurs numériques; garde les autres colonnes numériques (remainder = "passthrough")
transformation = ColumnTransformer(transformers= [("genre", OneHotEncoder(handle_unknown = "ignore"), ["track_genre"])], remainder = "passthrough")

#Relier la transformation au modèle de forêt aléatoire
modele = make_pipeline(transformation, RandomForestRegressor(n_estimators=100, random_state=42, n_jobs = -1))

# #Entraîner le modèle avec les données d'entraînement
modele.fit(x_train, y_train)

#Prédire la popularité des chansons du test
y_prediction = modele.predict(x_test)

#Évaluer les prédictions avec la MAE
mae_modele = mean_absolute_error(y_test, y_prediction)

#Afficher et comparer les deux MAE
print("MAE de référence:", round(mae_reference,2))
print("MAE du modèle de prédiction", round(mae_modele,2))



#Création du graphique de comparaison "prédit contre réel"

plt.figure(figsize=(7, 7))
plt.scatter(y_test, y_prediction, alpha=0.3)

# Ligne correspondant à des prédictions parfaites
plt.plot([0, 100], [0, 100], "r--", label="Prédiction parfaite")

plt.xlabel("Popularité réelle")
plt.ylabel("Popularité prédite")
plt.title("Popularité réelle et prédite")
plt.xlim(0, 100)
plt.ylim(0, 100)
plt.legend()
plt.show()


#Volet 2.6: Sauvegarde
chemin_modele = os.path.join("modeles", "modele_popularite.joblib")
chemin_metadonnees = os.path.join("modeles", "metadonnees.json")

#Sauvegarder le pipeline entraîné
joblib.dump(modele, chemin_modele, compress=3)

# Préparer les métadonnées à partir des données d'entraînement
colonnes_numeriques = x_train.select_dtypes(include="number").columns

metadonnees = {
    "cible": "popularity",
    "colonnes_entree": x_train.columns.tolist(),
    "genres_possibles": sorted(x_train["track_genre"].dropna().unique().tolist()),
    "intervalles_numeriques": {
        colonne: {
            "min": float(x_train[colonne].min()),
            "max": float(x_train[colonne].max())
        }
        for colonne in colonnes_numeriques
    },
    "metriques": {
        "MAE_modele": float(mae_modele),
        "MAE_reference": float(mae_reference)
    }
}

# Écrire les métadonnées dans le fichier JSON

with open(chemin_metadonnees,"w", encoding = "utf-8") as f:
    json.dump(metadonnees, f, ensure_ascii=False, indent=4)

#Pour acceder avec l'interface

# Récupérer le modèle Random Forest et le préprocesseur
random_forest = modele.named_steps["randomforestregressor"]
preprocesseur = modele.named_steps["columntransformer"]

# Récupérer les importances des caractéristiques
importances = random_forest.feature_importances_

# Récupérer les noms des colonnes après transformation
noms = preprocesseur.get_feature_names_out()

# Créer un dictionnaire pour regrouper les importances
importances_caracteristiques = {
    "Dansabilité": 0,
    "Énergie": 0,
    "Tempo": 0,
    "Volume": 0,
    "Durée": 0,
    "Genre musical": 0
}

for nom, importance in zip(noms, importances):

    if "genre__track_genre" in nom:
        importances_caracteristiques["Genre musical"] += importance

    elif "remainder__danceability" in nom:
        importances_caracteristiques["Dansabilité"] = importance

    elif "remainder__energy" in nom:
        importances_caracteristiques["Énergie"] = importance

    elif "remainder__tempo" in nom:
        importances_caracteristiques["Tempo"] = importance

    elif "remainder__loudness" in nom:
        importances_caracteristiques["Volume"] = importance

    elif "remainder__duration_ms" in nom:
        importances_caracteristiques["Durée"] = importance