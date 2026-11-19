import sys
from pathlib import Path

# Ajouter la racine du projet au chemin de recherche de Python
dossier_projet = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(dossier_projet))

from src.interface import afficher_interface


if __name__ == "__main__":
    afficher_interface()