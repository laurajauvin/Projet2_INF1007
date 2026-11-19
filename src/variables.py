import math
from abc import ABC, abstractmethod
from .exceptions import ErreurValidation

class Variable(ABC):
    """Classe parente des variables que l'interface doit valider."""

    def __init__(self, nom: str)-> None:
        if not nom.strip():
            raise ErreurValidation("Le nom d'une variable ne peut pas être vide")
        self._nom = nom

    @abstractmethod
    def valider(self, valeur:object)-> None:
         """Valide une valeur; les classes enfants définissent les règles."""
         raise NotImplementedError

    def __str__(self) -> str:
        """Renvoie le nom de la variable pour l'afficher sous forme de texte."""
        return self._nom

class VariableNumerique(Variable):
    """Variable dont la valeur doit être numérique et respecter un intervalle."""

    def __init__(self, nom: str, minimum: float, maximum: float) -> None:
        """Reçoit un nom et un intervalle; échoue si l'intervalle est inversé."""
        super().__init__(nom)

        if minimum > maximum:
            raise ErreurValidation("Le minimum ne peut pas dépasser le maximum.")

        self._minimum = minimum
        self._maximum = maximum

    def valider(self, valeur: object)-> None:
        """Vérifie que la valeur est un nombre fini dans l'intervalle permis."""
        #Vérifie que la valeur reçue n'est pas True ou False et est bien numérique (si True ou False, bool donne True et une erreur est levée)
        if isinstance(valeur, bool) or not isinstance(valeur, (int, float)): #vérifie si valeur est un entier ou un nombre décimal
            raise ErreurValidation(f"{self._nom} doit être un nombre.")
        
        if not math.isfinite(valeur):
            raise ErreurValidation(f"{self._nom} doit être une valeur finie.")

        if not self._minimum <= valeur <= self._maximum:
            raise ErreurValidation(
                f"{self._nom} doit être entre "
                f"{self._minimum} et {self._maximum}."
            )

class VariableCategorielle(Variable):
    """Variable dont la valeur doit appartenir à une liste autorisée."""

    def __init__(self, nom:str, valeurs_possibles: list[str]) -> None:
        """Reçoit un nom et les choix permis; échoue si la liste est vide."""
        super().__init__(nom)

        if not valeurs_possibles:
            raise ErreurValidation("La liste des valeurs possibles est vide.")
        self._valeurs_possibles = tuple(valeurs_possibles)
    
    def valider(self, valeur: object) -> None:
        """Vérifie que la valeur est une chaîne autorisée."""
        if not isinstance(valeur, str):
            raise ErreurValidation(f"{self._nom} doit être du texte.")

        if valeur not in self._valeurs_possibles:
            raise ErreurValidation(f"{valeur} n'est pas une valeur autorisée pour {self._nom}.")
    
    
        

        