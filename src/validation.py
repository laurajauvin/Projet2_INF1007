import json
import os

from .exceptions import ErreurValidation
from .variables import (
    VariableNumerique,
    VariableCategorielle,
    ValidateurVariables,
)


def creer_validateur():
    """Charge les métadonnées et crée le validateur."""

    dossier_src = os.path.dirname(os.path.abspath(__file__))
    dossier_projet = os.path.dirname(dossier_src)

    chemin_json = os.path.join(
        dossier_projet,
        "modeles",
        "metadonnees.json"
    )
    
    if not os.path.isfile(chemin_json):
        raise ErreurValidation(
        "Le fichier des métadonnées est introuvable."
        )

    try:
        with open(chemin_json, "r", encoding="utf-8") as fichier:
            metadonnees = json.load(fichier)

    except (json.JSONDecodeError, OSError, UnicodeDecodeError) as erreur:
        raise ErreurValidation(
            "Le fichier des métadonnées est invalide ou impossible à lire."
        ) from erreur

    variables = []

    for nom, intervalle in metadonnees["intervalles_numeriques"].items():
        variables.append(
            VariableNumerique(
                nom,
                intervalle["min"],
                intervalle["max"]
            )
        )

    variables.append(
        VariableCategorielle(
            "track_genre",
            metadonnees["genres_possibles"]
        )
    )

    return ValidateurVariables(variables)


def valider_entrees(
    chanson: str,
    artiste: str,
    genre: str,
    minutes: int,
    secondes: int,
    valeurs: dict[str, object],
    validateur: ValidateurVariables | None = None
) -> None:
    """Vérifie la validité des entrées de l'utilisateur."""

    if validateur is None:
        validateur = creer_validateur()

    # Vérifier le titre
    if not chanson.strip():
        raise ErreurValidation("Veuillez écrire le titre de la chanson.")

    # Vérifier l'artiste
    if not artiste.strip():
        raise ErreurValidation("Veuillez écrire le nom de l'artiste.")

    # Vérifier les minutes
    if minutes < 0:
        raise ErreurValidation("Les minutes ne peuvent pas être négatives.")

    # Vérifier les secondes
    if not 0 <= secondes <= 59:
        raise ErreurValidation("Les secondes doivent être entre 0 et 59.")

    # Regrouper toutes les valeurs à valider
    valeurs_completes = valeurs.copy()
    valeurs_completes["track_genre"] = genre

    # Valider les caractéristiques numériques et le genre
    validateur.valider_tout(valeurs_completes)
