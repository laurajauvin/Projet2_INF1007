#Exception personnalisée qui permet de signaler clairement qu'une saisie de l'utilisateur est invalide
class ErreurValidation(Exception):
    """Erreur déclenchée lorsqu'une entrée utilisateur est invalide."""
    pass