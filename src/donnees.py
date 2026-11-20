import os
import pandas as pd
import matplotlib.pyplot as plt

# Construire les chemins à partir de l'emplacement de ce fichier
dossier_src = os.path.dirname(os.path.abspath(__file__)) #__file__ désigne le fichier python actuel (donnees.py)
dossier_projet = os.path.dirname(dossier_src) #prend dossier contenant le fichier

dossier_donnees = os.path.join(dossier_projet, "donnees")


def charger_donnees(nom_fichier: str) -> pd.DataFrame: #permet utilisation fonctions pandas (choisir un sample, nettoyer données...)
    """Charge un CSV.
    Reçoit le chemin du fichier, lit le fichier et renvoie son contenu en DataFrame.
    Lève FileNotFoundError si le fichier n'existe pas.
    """
    chemin = os.path.join(dossier_donnees, nom_fichier)
    if not os.path.isfile(chemin):
        raise FileNotFoundError(f"Fichier introuvable : {chemin}")

    with open(chemin, "r", encoding="utf-8") as fichier:
        return pd.read_csv(fichier)



#Empêche l'exécution du sample et nettoyage de données à moins que soit exécuté à partir du fichier donnees.py;
#le sampling ne se refait pas à chaque run
if __name__ == "__main__": 

    # 1. Charger le dataset original
    df = charger_donnees("dataset.csv")

    # 2. Sélectionner aléatoirement 30 000 chansons
    df_30k = df.sample(n=30000, random_state=42)

    # Enregistrer le nouveau dataset
    chemin_echantillon = os.path.join(dossier_donnees, "dataset_30000.csv")

    with open(chemin_echantillon, "w", encoding="utf-8", newline = "") as fichier:
        df_30k.to_csv(fichier, index = False) #écrit le DataFrame dans un fichier CSV

    #On met l'échantillon enregistré dans df -> c'est ce qui sera exploré et nettoyé
    df = charger_donnees("dataset_30000.csv")


    # 3. Explorer le nouveau dataset
   
    #Affiche les noms des colonnes, leur type et le nombre de valeurs non manquantes
    print("\n--- Informations générales ---")
    df.info() #(pas de print, car sinon Python affiche None après affichage de l'info)

    #Affiche des statistiques descriptives pour les colonnes numériques : moyenne, minimum, maximum, quartiles, etc.
    print("\n--- Statistiques descriptives ---")
    print(df.describe())

    #Compte et affiche le nombre de valeurs manquantes de chaque colonne (tableau de True et False, sum compte les True)
    print("\n--- Valeurs manquantes ---")
    print(df.isnull().sum()) #(nous avons toutefois gardé ces valeurs manquantes, car elles ne servaient pas à notre modèle
    #et les retirer aurait supprimé trois chansons pour rien)

    #Compte et affiche les lignes dupliquées, en double
    print("\n--- Doublons ---")
    print("Nombre de doublons :", df.duplicated().sum())


    # 4. Vérifier les valeurs suspectes
    print("\n--- Vérification des valeurs suspectes ---")


    #Exploration des chansons les plus longues
    print("\n--- Distribution des durées ---")
    print(df["duration_ms"].describe(percentiles=[0.01, 0.05, 0.95, 0.99])) #valeurs extrêmes de durée de chansons (ex. 0.95: 95% des chansons sont plus courtes ou égales)

   
    print("\nLes 10 durées les plus longues :")
    print((df["duration_ms"].nlargest(10) / 60000).round(2))

    print("Durées égales à 0 :", (df["duration_ms"] == 0).sum())
    print("Tempos égaux à 0 :", (df["tempo"] == 0).sum())

    #Sert à repérer des valeurs de volume inhabituelles et à décider si elles méritent d'être examinées
    print("\n--- Distribution du volume ---")
    print(df["loudness"].describe( percentiles=[0.01, 0.05, 0.95, 0.99]))



    #Liste contenant uniquement les variables ayant des valeurs entre 0 et 1
    colonnes_01 = [
        "danceability",
        "energy",
        "speechiness",
        "acousticness",
        "instrumentalness",
        "liveness",
        "valence"
    ]

    #Vérifie et compte quelles valeurs sont sous 0 OU au-dessus de 1
    for colonne in colonnes_01:
        print(
            colonne,
            "hors intervalle [0, 1] :",
            ((df[colonne] < 0) | (df[colonne] > 1)).sum()) 


    # 5. Nettoyer les données

    # Supprimer la colonne d'index inutile
    df = df.drop(columns=["Unnamed: 0"])

    # Seulement garder les chansons ayant une durée ET un tempo strictement supérieurs à 0
    df = df[
        (df["duration_ms"] > 0) &
        (df["tempo"] > 0)
    ]

    # Vérifier que tout a bien été filtré
    print("\n--- Après le nettoyage ---")
    print("Nombre de chansons restantes :", len(df)) #29948

    print("Durées égales à 0 :",
        (df["duration_ms"] == 0).sum())

    print("Tempos égaux à 0 :",
        (df["tempo"] == 0).sum())

    # Enregistrer le dataset nettoyé
    chemin_nettoye = os.path.join(dossier_donnees, "dataset_30000_clean.csv")
    with open(chemin_nettoye, "w", encoding = "utf-8", newline = "") as fichier:
        df.to_csv(fichier, index=False)


    #Regroupement des colonnes utilisées par notre modèle ainsi que la variable cible
    variables = [
        "danceability",
        "energy",
        "tempo",
        "loudness",
        "duration_ms",
        "track_genre",
        "popularity"
    ]

    #S'assurer qu'il n'y a pas de valeurs manquantes dans ces colonnes
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

