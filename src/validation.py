import json
import os
from src.exceptions import ErreurValidation
from src.variables import (
    VariableNumerique,
    VariableCategorielle,
    ValidateurVariables,
)


def creer_validateur() -> ValidateurVariables:
    
    """Crée un validateur à partir des métadonnées du modèle.

    Ne reçoit aucun argument.
    Renvoie un objet ValidateurVariables contenant les règles de validation.
    Lève ErreurValidation si le fichier de métadonnées est introuvable
    ou si son contenu ne peut pas être lu correctement.
    """

    # ---- Repérer le dossier du fichier actuel, puis remonter au dossier principal du projet ----

    dossier_src = os.path.dirname(os.path.abspath(__file__))
    dossier_projet = os.path.dirname(dossier_src)

    # Créer chemin vers le fichier JSON
    chemin_json = os.path.join(
        dossier_projet,
        "modeles",
        "metadonnees.json"
    )


    # ---- Empêcher la création du validateur si le fichier JSON est absent ----

    if not os.path.isfile(chemin_json):
        raise ErreurValidation(
        "Le fichier des métadonnées est introuvable."
        )

    try:
        # Ouvrir le fichier en lecture et charger son contenu
        with open(chemin_json, "r", encoding="utf-8") as fichier: 
            metadonnees = json.load(fichier)

    # ErreurValidation pour de gérer les erreurs plus facilement et donner un message plus clair à l'utilisateur
    except (json.JSONDecodeError, OSError, UnicodeDecodeError) as erreur: # Format incorrect, problème d'accès, mauvais encodage
        raise ErreurValidation(    # ErreurValidation = exception personnalisée (fichier exceptions)
            "Le fichier des métadonnées est invalide ou impossible à lire."
        ) from erreur  # Conserve lien avec erreur originale pour pouvoir retrouver sa cause


    # ---- Préparer une liste qui contiendra tous les validateurs individuels ----

    variables = []

    # Parcourir les caractéristiques numériques et récupérer limites min et max avec JSON
    for nom, intervalle in metadonnees["intervalles_numeriques"].items():
        variables.append(
            VariableNumerique(
                nom,
                intervalle["min"],
                intervalle["max"]
            )
        )

    # Ajout d'un validateur à la liste
    variables.append(
        VariableCategorielle(
            "track_genre",
            metadonnees["genres_possibles"]
        )
    )

    return ValidateurVariables(variables) # Fonction creer_validateur() retourne un objet regroupant tous les validateurs


# ---- Sert à vérifier si les entrées de l'utilisateur sont valides ----

def valider_entrees(
    chanson: str,
    artiste: str,
    genre: str,
    minutes: int,
    secondes: int,
    valeurs: dict[str, object],
    validateur: ValidateurVariables | None = None) -> None:

    """Vérifie les informations saisies pour une chanson.

    Reçoit le titre, l'artiste, le genre, les caractéristiques numériques
    et le validateur à utiliser.
    Ne renvoie aucune valeur si toutes les entrées sont valides.
    Lève ErreurValidation si une entrée est vide ou ne respecte pas
    les règles de validation.
    """

    # Si aucun validateur n'a été fourni, en créer un à partir du JSON
    if validateur is None:
        validateur = creer_validateur()

    # Refuser titre vide ou composé uniquement d'espaces
    if not chanson.strip():
        raise ErreurValidation("Veuillez écrire le titre de la chanson.")

    # Refuser nom d'artiste vide ou composé uniquement d'espaces
    if not artiste.strip():
        raise ErreurValidation("Veuillez écrire le nom de l'artiste.")

    # Refuser minutes négatives
    if minutes < 0:
        raise ErreurValidation("Les minutes ne peuvent pas être négatives.")

    # Refuser seconde hors de 0 à 59
    if not 0 <= secondes <= 59:
        raise ErreurValidation("Les secondes doivent être entre 0 et 59.")

    # Regrouper toutes les valeurs à valider
    valeurs_completes = valeurs.copy()   # Copier caractéristiques musicales pour pas modifie dictionnaire original de l'interface
    valeurs_completes["track_genre"] = genre   # Ajoute le genre dans les variables à valider

    # Valider les caractéristiques numériques et le genre
    validateur.valider_tout(valeurs_completes) # Erreur dès qu'une caractéristique est invalide
