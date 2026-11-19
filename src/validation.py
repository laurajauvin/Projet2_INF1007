import json
import math
from pathlib import Path
import os


class ErreurValidation(Exception):
    """Erreur déclenchée lorsqu'une entrée utilisateur est invalide."""
    pass


# Charger les métadonnées
dossier_src = os.path.dirname(os.path.abspath(__file__))
dossier_projet = os.path.dirname(dossier_src)

chemin_json = os.path.join(
    dossier_projet,
    "modeles",
    "metadonnees.json"
)

with open(chemin_json, "r", encoding="utf-8") as fichier:
    metadonnees = json.load(fichier)


def valider_entrees(
    chanson: str,
    artiste: str,
    genre: str,
    valeurs: dict[str, int | float]
) -> None:
    """
    Vérifie la validité des entrées de l'utilisateur.

    Args:
        chanson: Titre de la chanson.
        artiste: Nom de l'artiste.
        genre: Genre musical sélectionné.
        valeurs: Caractéristiques numériques de la chanson.

    Raises:
        ErreurValidation: Si une entrée est invalide.
    """

    # Vérifier le titre
    if not chanson.strip():
        raise ErreurValidation("Veuillez écrire le titre de la chanson.")

    # Vérifier l'artiste
    if not artiste.strip():
        raise ErreurValidation("Veuillez écrire le nom de l'artiste.")

    # Vérifier le genre
    if genre not in metadonnees["genres_possibles"]:
        raise ErreurValidation("Veuillez sélectionner un genre valide.")

    # Vérifier les caractéristiques numériques
    intervalles = metadonnees["intervalles_numeriques"]

    for caracteristique, valeur in valeurs.items():

        minimum = intervalles[caracteristique]["min"]
        maximum = intervalles[caracteristique]["max"]

        # Vérifier que la valeur est numérique
        if isinstance(valeur, bool) or not isinstance(valeur, (int, float)):
            raise ErreurValidation(
                f"La valeur de {caracteristique} doit être numérique."
            )

        # Vérifier que la valeur est finie
        if not math.isfinite(valeur):
            raise ErreurValidation(
                f"La valeur de {caracteristique} est invalide."
            )

        # Vérifier que la valeur respecte l'intervalle
        if not minimum <= valeur <= maximum:

            if caracteristique == "duration_ms":
                minimum_min = int(minimum // 60000)
                minimum_sec = int((minimum % 60000) // 1000)

                maximum_min = int(maximum // 60000)
                maximum_sec = int((maximum % 60000) // 1000)

                raise ErreurValidation(
                    f"La durée doit être entre "
                    f"{minimum_min} min {minimum_sec} s et "
                    f"{maximum_min} min {maximum_sec} s."
                )
             
            else: 
                raise ErreurValidation(
                    f"{caracteristique} doit être entre {minimum} et {maximum}."
                )

