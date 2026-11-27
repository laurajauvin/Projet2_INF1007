import sys
from pathlib import Path
from src.interface import afficher_interface

#Point d'entrée du projet
# Ajouter la racine du projet au chemin de recherche de Python
#pour pouvoir importer l'interface depuis src
dossier_projet = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(dossier_projet))


#Quand on lance main.py, il appelle afficher_interface() pour démarrer l'application
#interface est assez car regroupe toutes les fonctions nécessaires au modèle et à son affichage
if __name__ == "__main__":
    afficher_interface()