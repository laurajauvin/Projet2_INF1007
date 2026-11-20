import json
import os
from .exceptions import ErreurValidation
from .variables import (
    VariableNumerique,
    VariableCategorielle,
    ValidateurVariables)

def creer_validateur():
    """Charge les métadonnées et crée le validateur."""

    dossier_src = os.path.dirname(os.path.abspath(__file__))
    dossier_projet = os.path.dirname(dossier_src)

    chemin_json = os.path.join(
        dossier_projet,
        "modeles",
        "metadonnees.json"
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
            "genre",
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
    valeurs: dict[str, int | float]
) -> None:
    """Vérifie la validité des entrées de l'utilisateur."""

    # Vérifier le titre
    if not chanson.strip():
        raise ErreurValidation("Veuillez écrire le titre de la chanson.")

    # Vérifier l'artiste
    if not artiste.strip():
        raise ErreurValidation("Veuillez écrire le nom de l'artiste.")

    # Vérifier les minutes:
    if minutes < 0:
        raise ErreurValidation("Les minutes ne peuvent pas être négatives.")

    # Vérifier les secondes
    if not 0 <= secondes <= 59:
        raise ErreurValidation("Les secondes doivent être entre 0 et 59.")

    # Regrouper toutes les valeurs à valider
    valeurs_completes = valeurs.copy()
    valeurs_completes["genre"] = genre

    # Valider les caractéristiques numériques et le genre
    validateur = creer_validateur()
    validateur.valider_tout(valeurs_completes)
