import os
import pandas as pd
import matplotlib.pyplot as plt
from .exceptions import DonneeInvalideError, GenreInconnuError

dossier_src = os.path.dirname(os.path.abspath(__file__))
dossier_projet = os.path.dirname(dossier_src)

dossier_donnees = os.path.join(dossier_projet, "donnees")


def charger_donnees(nom_fichier: str) -> pd.DataFrame:
    """Charge un CSV.
    Reçoit le chemin du fichier et renvoie son contenu en DataFrame.
    Lève FileNotFoundError si le fichier n'existe pas.
    """
    chemin = os.path.join(dossier_donnees, nom_fichier)
    if not os.path.isfile(chemin):
        raise FileNotFoundError(f"Fichier introuvable : {chemin}")

    with open(chemin, "r", encoding="utf-8") as fichier:
        return pd.read_csv(fichier)




if __name__ == "__main__": #empêche l'exécution du sample et nettoyage de données à moins que soit exécuté à partir du fichier donnees.py

    # 1. Charger le dataset original
    df = charger_donnees("dataset.csv")

    print(df.head())
    print("Nombre de lignes :", len(df))


    # 2. Sélectionner aléatoirement 30 000 chansons
    df_30k = df.sample(n=30000, random_state=42)

    # Enregistrer le nouveau dataset
    chemin_echantillon = os.path.join(dossier_donnees, "dataset_30000.csv")

    with open(chemin_echantillon, "w", encoding="utf-8", newline = "") as fichier:
        df_30k.to_csv(fichier, index = False) #écrit le DataFrame dans un fichier CSv

    df = charger_donnees("dataset_30000.csv")

    print("Nombre de chansons sélectionnées :", len(df_30k))


    # 3. Explorer le nouveau dataset
   
    print("\n--- Informations générales ---")
    df.info() #(pas de print, car sinon Python affiche None après affichage de l'info)

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
    chemin_nettoye = os.path.join(dossier_donnees, "dataset_30000_clean.csv")

    with open(chemin_nettoye, "w", encoding = "utf-8", newline = "") as fichier:
        df.to_csv(fichier, index=False)

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
    chemin_graphique = os.path.join(dossier_projet, "distribution_popularite.png")
    plt.savefig(chemin_graphique)

    # Afficher le graphique
    plt.show()


