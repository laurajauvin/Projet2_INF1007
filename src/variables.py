import math
from abc import ABC, abstractmethod
from .exceptions import ErreurValidation

class Variable(ABC):
    """Classe parente des variables que l'interface doit valider."""

    def __init__(self, nom: str)-> None:
        """Reçoit le nom interne de la variable.
        Lève ErreurValidation si le nom est vide.
        """
        if not nom.strip():
            raise ErreurValidation("Le nom d'une variable ne peut pas être vide")
        self._nom = nom #._ indique que cet attribut est destiné à être utilisé à l'intérieur de sa classe et de ses classes enfants

    @abstractmethod #chaque classe enfant doit définir sa propre façon de valider une valeur
    def valider(self, valeur:object)-> None:
         """Valide une valeur; les classes enfants définissent les règles."""
         raise NotImplementedError #empêche de créer une classe qui ne redéfinit pas cette méthode

    #permet de fournir le nom à une autre partie du programme qui l'appelle
    def __str__(self) -> str:
        """Renvoie le nom de la variable pour l'afficher sous forme de texte."""
        return self._nom

    #Permet de consulter le nom avec variable.nom sans accéder directement à l'attribut interne _nom
    @property #(pas de setter car on ne modifie pas le nom)
    def nom(self) -> str:
        """Renvoie le nom de la variable."""
 
        return self._nom

#classe enfant de Variable
class VariableNumerique(Variable):
    """Variable dont la valeur doit être numérique et respecter un intervalle."""

    def __init__(self, nom: str, minimum: float, maximum: float) -> None:
        """Reçoit un nom et un intervalle; échoue si l'intervalle est inversé."""
        super().__init__(nom) #appelle le constructeur de Variable 

        if minimum > maximum:
            raise ErreurValidation("Le minimum ne peut pas dépasser le maximum.")

        self.__minimum = minimum #double underscore: privés à la classe -> évite qu'une autre partie du code ne la modifie
        self.__maximum = maximum

    def valider(self, valeur: object)-> None:
        """Vérifie que la valeur est un nombre fini dans l'intervalle permis."""
        #Vérifie que la valeur reçue n'est pas True ou False et est bien numérique (si True ou False, bool donne True et une erreur est levée)
        if isinstance(valeur, bool) or not isinstance(valeur, (int, float)): #vérifie si valeur est un entier ou un nombre décimal
            raise ErreurValidation(f"{self._nom} doit être un nombre.")
        
        if not math.isfinite(valeur):
            raise ErreurValidation(f"{self._nom} doit être une valeur finie.")

        if not self.__minimum <= valeur <= self.__maximum:
            if self._nom == "duration_ms":
                minimum = math.ceil(self.__minimum / 1000) #arrondit borne min vers le haut
                maximum = math.floor(self.__maximum / 1000) #arrondit borne max vers le bas


                raise ErreurValidation(
                    f"La durée doit être entre "
                    f"{minimum // 60} min {minimum % 60} s et "
                    f"{maximum // 60} min {maximum % 60} s."
                )

            raise ErreurValidation(
                f"{self._nom} doit être entre "
                f"{self.__minimum} et {self.__maximum}."
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
    
    
class ValidateurVariables: #rassemble plusieurs objets Variable pour valider toutes les caractéristiques de la chanson
    """Regroupe les variables à valider."""

    def __init__(self, variables: list[Variable]) -> None:
        """Reçoit et conserve la liste des variables à valider."""
        self._variables = variables

    def valider_tout(self, valeurs: dict[str, object]) -> None:
        """Valide chaque valeur entrée par l'utilisateur associée à une variable.
        Reçoit un dictionnaire indexé par les noms des variables.
        Lève ErreurValidation si une valeur est invalide.
        """
        for variable in self._variables:
            if variable.nom not in valeurs:
                raise ErreurValidation(f"La valeur de {variable.nom} est manquante.")

            #appelle la version de valider qui correspond au type réel de variable
            variable.valider(valeurs[variable.nom]) 