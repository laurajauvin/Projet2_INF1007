class DonneeInvalideError(Exception):
    """Erreur pour une valeur musicale invalide."""


class GenreInconnuError(DonneeInvalideError):
    """Erreur pour un genre absent de la liste autorisée."""


