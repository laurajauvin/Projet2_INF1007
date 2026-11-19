import os
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.ensemble import RandomForestRegressor
from donnees import charger_donnees


#Volet 2.4: Établir une référence
def calculer_reference(y_train : pd.Series,y_test : pd.Series) -> float:
    """Calcule la MAE d'une référence qui prédit toujours la moyenne de y_train.
    Reçoit les popularités d'entraînement et de test.
    Renvoie un nombre : la MAE de référence.
    Lève ValueError si l'un des deux ensembles est vide.
    """
    if y_train.empty or y_test.empty:
        raise ValueError("Les ensembles train et test ne doivent pas être vides.")
    
    #Calcule la popularité moyenne des données d'entraînement
    moyenne = float(y_train.mean())
    #Crée une prédiction pour chaque chanson de test; elles ont toutes la même valeur: la moyenne calculée juste avant
    y_reference = [moyenne] * len(y_test)
    
    #Calcul de la métrique d'évaluation MAE
    #Compare les prédictions aux vraies popularités dans y_test
    return float(mean_absolute_error(y_test, y_reference))






#Volet 2.5: Entraîner et évaluer
def creer_modele() -> Pipeline:
    """Construit le pipeline d'encodage de prédiction de la popularité."""

    #Transformation de track_genre en valeurs numériques; garde les autres colonnes numériques (remainder = "passthrough")
    transformation = ColumnTransformer( transformers=[("genre", OneHotEncoder(handle_unknown="ignore"), ["track_genre"],)], remainder="passthrough")
    
    #Relier la transformation au modèle de forêt aléatoire
    return make_pipeline(transformation, RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1))




def entrainer_modele(modele: Pipeline,x_train: pd.DataFrame,y_train: pd.Series) -> Pipeline:
    """Entraîne le modèle sur les données d'entraînement et le renvoie."""
    modele.fit(x_train, y_train)
    return modele


def predire(modele: Pipeline,x_test: pd.DataFrame) -> np.ndarray:

    """Renvoie les popularités prédites pour les chansons de test."""
    return modele.predict(x_test)

def calculer_mae(y_test: pd.Series,y_prediction: np.ndarray) -> float:
    
    """Calcule la MAE entre les valeurs réelles du test et les prédictions."""
    return float(mean_absolute_error(y_test, y_prediction))


def creer_graphique(y_test: pd.Series, y_prediction: np.ndarray,) -> Figure:
    """Crée et renvoie le graphique de comparaison des popularités réelles et prédites."""
    figure = plt.figure(figsize=(7, 7))

    plt.scatter(y_test, y_prediction, alpha=0.3)
    # Ligne correspondant à des prédictions parfaites
    plt.plot([0, 100], [0, 100], "r--", label="Prédiction parfaite")
    plt.xlabel("Popularité réelle")
    plt.ylabel("Popularité prédite")
    plt.title("Popularité réelle et prédite")
    plt.xlim(0, 100)
    plt.ylim(0, 100)
    plt.legend()

    return figure

#Pour acceder avec l'interface
def calculer_importances(modele: Pipeline) -> dict[str, float]:
    """Regroupe les importances par caractéristique d'origine.
    Reçoit le pipeline déjà entraîné et renvoie les importances regroupées.
    Le pipeline doit contenir les étapes 'randomforestregressor' et
    'columntransformer' sinon, l'accès aux étapes ou aux importances échoue.
    """
    random_forest = modele.named_steps["randomforestregressor"]
    preprocesseur = modele.named_steps["columntransformer"]

    # Récupérer les importances des caractéristiques
    importances = random_forest.feature_importances_

    # Récupérer les noms des colonnes après transformation
    noms = preprocesseur.get_feature_names_out()

    # Créer un dictionnaire pour regrouper les importances
    importances_caracteristiques = {
        "Dansabilité": 0.0,
        "Énergie": 0.0,
        "Tempo": 0.0,
        "Volume": 0.0,
        "Durée": 0.0,
        "Genre musical": 0.0,
    }

    for nom, importance in zip(noms, importances):
        if "genre__track_genre" in nom:
            importances_caracteristiques["Genre musical"] += float(importance)
        elif "remainder__danceability" in nom:
            importances_caracteristiques["Dansabilité"] = float(importance)
        elif "remainder__energy" in nom:
            importances_caracteristiques["Énergie"] = float(importance)
        elif "remainder__tempo" in nom:
            importances_caracteristiques["Tempo"] = float(importance)
        elif "remainder__loudness" in nom:
            importances_caracteristiques["Volume"] = float(importance)
        elif "remainder__duration_ms" in nom:
            importances_caracteristiques["Durée"] = float(importance)

    return importances_caracteristiques


def creer_metadonnees(x_train: pd.DataFrame,mae_modele: float,mae_reference: float,importances: dict[str, float]) -> dict[str, object]:
    
    """Prépare les colonnes, intervalles et résultats à écrire dans le JSON."""

    # Préparer les métadonnées à partir des données d'entraînement
    colonnes_numeriques = x_train.select_dtypes(include="number").columns

    return {"cible": "popularity","colonnes_entree": x_train.columns.tolist(),"genres_possibles": sorted(x_train["track_genre"].dropna().unique().tolist()),
            "intervalles_numeriques": {colonne: {
                "min": float(x_train[colonne].min()),
                "max": float(x_train[colonne].max()),
                }
            for colonne in colonnes_numeriques},
        "metriques": {
            "MAE_modele": mae_modele,
            "MAE_reference": mae_reference,
        },
        "importance_caracteristiques": importances,
    }


def sauvegarder_resultats(modele: Pipeline,metadonnees: dict[str, object],chemin_modele: str,chemin_json: str) -> None:
   
    """Sauvegarde le modèle compressé et les métadonnées JSON."""
    os.makedirs(os.path.dirname(chemin_modele), exist_ok=True)

    with open(chemin_modele, "wb") as fichier:
        joblib.dump(modele, fichier, compress=3)

    with open(chemin_json, "w", encoding="utf-8") as fichier:
        json.dump(metadonnees, fichier, ensure_ascii=False, indent=4)




if __name__ == "__main__":
    df = charger_donnees("dataset_30000_clean.csv")

    # Séparation des données
    x = df[["danceability", "energy", "tempo", "loudness",
            "duration_ms", "track_genre"]]
    y = df["popularity"]

    #Séparer les données de test et d'entraînement
    #Environ 80% des lignes vont à l'entraînement et 20% au test; random = 42 permet d'obtenir la même séparation à chaque fois qu'on relance le code
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    #Référence
    mae_reference = calculer_reference(y_train, y_test)

    #Entraînement et prédictions
    modele = creer_modele()
    modele = entrainer_modele(modele, x_train, y_train)
    y_prediction = predire(modele, x_test)
    mae_modele = calculer_mae(y_test, y_prediction)

    print("MAE de référence :", round(mae_reference, 2))
    print("MAE du modèle :", round(mae_modele, 2))

    # Graphique réel contre prédit
    figure = creer_graphique(y_test, y_prediction)
    plt.show()

    #Importances et métadonnées
    importances = calculer_importances(modele)
    metadonnees = creer_metadonnees(x_train,mae_modele,mae_reference,importances)

    #Chemins et sauvegarde des fichiers
    dossier_src = os.path.dirname(os.path.abspath(__file__))
    dossier_projet = os.path.dirname(dossier_src)
    chemin_modele = os.path.join(dossier_projet, "modeles", "modele_popularite.joblib")
    chemin_metadonnees = os.path.join(dossier_projet, "modeles", "metadonnees.json")

    sauvegarder_resultats(modele,metadonnees,chemin_modele,chemin_metadonnees)




