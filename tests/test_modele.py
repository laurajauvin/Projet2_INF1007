import json
import joblib
import numpy as np
import pandas as pd
import pytest

from src.donnees import charger_donnees
from src.entrainement import (
    calculer_importances,
    calculer_mae,
    calculer_mae_numpy,
    calculer_reference,
    creer_metadonnees,
    creer_modele,
    entrainer_modele,
    predire,
    sauvegarder_resultats,
)
from src.exceptions import ErreurValidation
from src.validation import creer_validateur, valider_entrees
from src.variables import VariableNumerique


# Colonnes que le modèle utilise comme entrée
COLONNES_ENTREE = [
    "danceability",
    "energy",
    "tempo",
    "loudness",
    "duration_ms",
    "track_genre",
]


# Fixture : charge le dataset nettoyé et choisit un échantillon de 60 lignes
@pytest.fixture(scope="module")
def df_test() -> pd.DataFrame:
    """Charge le CSV réel et en prend 60 lignes pour les tests."""
    df = charger_donnees("dataset_30000_clean.csv")
    return df.sample(n=60, random_state=42)


# Fixture : entraîne le modèle une seule fois pour les tests
@pytest.fixture(scope="module")
def modele_entraine(df_test: pd.DataFrame):
    """Entraîne le modèle sur l'échantillon utilisé dans les tests."""
    return entrainer_modele(
        creer_modele(),
        df_test[COLONNES_ENTREE],
        df_test["popularity"],
    )


# Fixture : crée le validateur une seule fois pour les tests
@pytest.fixture(scope="module")
def validateur():
    """Crée le validateur pour les tests de validation."""
    return creer_validateur()


# ------ Tests de calcul de la MAE ------

def test_reference_calcule_mae() -> None:
    """Vérifie la MAE de référence sur un exemple calculable à la main."""
    y_train = pd.Series([10, 20])
    y_test = pd.Series([10, 30])

    resultat = calculer_reference(y_train, y_test)

    assert resultat == pytest.approx(10.0)


def test_mae_numpy_identique_a_sklearn() -> None:
    """Vérifie que le calcul NumPy donne le même résultat que scikit-learn."""
    y_test = pd.Series([10, 20, 30, 40])
    y_prediction = np.array([12, 18, 25, 45])

    resultat_numpy = calculer_mae_numpy(y_test, y_prediction)
    resultat_sklearn = calculer_mae(y_test, y_prediction)

    assert resultat_numpy == pytest.approx(resultat_sklearn)


def test_reference_refuse_train_vide() -> None:
    """Vérifie qu'une erreur est levée si y_train est vide."""
    with pytest.raises(ValueError):
        calculer_reference(
            pd.Series(dtype=float),
            pd.Series([10, 20]),
        )


def test_reference_refuse_test_vide() -> None:
    """Vérifie qu'une erreur est levée si y_test est vide."""
    with pytest.raises(ValueError):
        calculer_reference(
            pd.Series([10, 20]),
            pd.Series(dtype=float),
        )


def test_mae_calcule_erreur_moyenne() -> None:
    """Vérifie la MAE avec deux valeurs réelles et prédites."""
    y_test = pd.Series([10, 20])
    y_prediction = np.array([12, 16])

    assert calculer_mae(y_test, y_prediction) == pytest.approx(3.0)


def test_mae_refuse_des_tailles_differentes() -> None:
    """Vérifie qu'une erreur est levée si les séries n'ont pas la même taille."""
    with pytest.raises(ValueError):
        calculer_mae(
            pd.Series([10, 20]),
            np.array([12]),
        )


# ------ Tests du modèle ------

def test_modele_contient_preprocesseur_et_foret() -> None:
    """Vérifie que le pipeline contient les deux étapes attendues."""
    modele = creer_modele()

    assert "columntransformer" in modele.named_steps
    assert "randomforestregressor" in modele.named_steps


def test_modele_entraine_produit_une_prediction_par_chanson(
    df_test: pd.DataFrame,
    modele_entraine,
) -> None:
    """Vérifie que le modèle prédit chaque ligne de l'échantillon."""
    predictions = predire(modele_entraine, df_test[COLONNES_ENTREE])

    assert len(predictions) == len(df_test)
    assert np.all(predictions >= 0)
    assert np.all(predictions <= 100)
    assert np.all(np.isfinite(predictions))


# ------ Tests des importances ------

def test_importances_sont_regroupees(modele_entraine) -> None:
    """Vérifie les noms des caractéristiques et la somme des importances."""
    importances = calculer_importances(modele_entraine)

    assert "Genre musical" in importances
    assert sum(importances.values()) == pytest.approx(1.0)
    assert len(importances) == 6


# ------ Tests des métadonnées ------

def test_metadonnees_contiennent_genres_et_intervalles(
    df_test: pd.DataFrame,
    modele_entraine,
) -> None:
    """Vérifie que les métadonnées reprennent le vrai échantillon."""
    importances = calculer_importances(modele_entraine)

    metadonnees = creer_metadonnees(
        df_test[COLONNES_ENTREE],
        mae_modele=12.0,
        mae_reference=19.0,
        importances=importances,
    )

    assert set(metadonnees["genres_possibles"]) == set(
        df_test["track_genre"].unique()
    )

    assert metadonnees["intervalles_numeriques"]["energy"]["min"] == 0.0
    assert metadonnees["intervalles_numeriques"]["energy"]["max"] == 1.0

    assert metadonnees["intervalles_numeriques"]["danceability"]["min"] == 0.0
    assert metadonnees["intervalles_numeriques"]["danceability"]["max"] == 1.0


# ------ Tests de sauvegarde ------

def test_sauvegarde_modele_et_metadonnees(
    tmp_path,
    modele_entraine,
) -> None:
    """Vérifie que le modèle et le JSON peuvent être enregistrés puis relus."""
    chemin_modele = str(tmp_path / "modeles" / "modele.joblib")
    chemin_json = str(tmp_path / "modeles" / "metadonnees.json")
    metadonnees = {"cible": "popularity"}

    sauvegarder_resultats(
        modele_entraine,
        metadonnees,
        chemin_modele,
        chemin_json,
    )

    assert joblib.load(chemin_modele) is not None

    with open(chemin_json, "r", encoding="utf-8") as fichier:
        assert json.load(fichier) == metadonnees


# ------ Tests de validation des entrées utilisateur ------


# Données valides utilisées comme base pour les tests

valeurs_valides = {
    "danceability": 0.5,
    "energy": 0.8,
    "tempo": 120.0,
    "loudness": -10.0,
    "duration_ms": 180000,
}

minutes_valides = 3
secondes_valides = 0


# Chanson valide passe toutes les vérifications sans déclencher d'exception

def test_entrees_valides(validateur):
    """Vérifie que des entrées valides sont acceptées."""
    valider_entrees(
        "Ma chanson",
        "Mon artiste",
        "pop",
        minutes_valides,
        secondes_valides,
        valeurs_valides,
        validateur,
    )


# Titre de chanson vide déclenche une erreur de validation

def test_titre_vide(validateur):
    """Vérifie qu'un titre vide est refusé."""
    with pytest.raises(ErreurValidation):
        valider_entrees(
            "",
            "Mon artiste",
            "pop",
            minutes_valides,
            secondes_valides,
            valeurs_valides,
            validateur,
        )


# Nom d'artiste vide déclenche une erreur de validation

def test_artiste_vide(validateur):
    """Vérifie qu'un artiste vide est refusé."""
    with pytest.raises(ErreurValidation):
        valider_entrees(
            "Ma chanson",
            "",
            "pop",
            minutes_valides,
            secondes_valides,
            valeurs_valides,
            validateur,
        )


# Genre absent de la liste déclenche une erreur de validation

def test_genre_invalide(validateur):
    """Vérifie qu'un genre inexistant est refusé."""
    with pytest.raises(ErreurValidation):
        valider_entrees(
            "Ma chanson",
            "Mon artiste",
            "jazz-invalide",
            minutes_valides,
            secondes_valides,
            valeurs_valides,
            validateur,
        )


# Dansabilité > 1.00 déclenche une erreur de validation

def test_dansabilite_invalide(validateur):
    """Vérifie qu'une dansabilité supérieure à 1 est refusée."""
    valeurs = valeurs_valides.copy()
    valeurs["danceability"] = 1.5

    with pytest.raises(ErreurValidation):
        valider_entrees(
            "Ma chanson",
            "Mon artiste",
            "pop",
            minutes_valides,
            secondes_valides,
            valeurs,
            validateur,
        )


# Énergie < 0.00 déclenche une erreur de validation

def test_energie_invalide(validateur):
    """Vérifie qu'une énergie inférieure à 0 est refusée."""
    valeurs = valeurs_valides.copy()
    valeurs["energy"] = -0.5

    with pytest.raises(ErreurValidation):
        valider_entrees(
            "Ma chanson",
            "Mon artiste",
            "pop",
            minutes_valides,
            secondes_valides,
            valeurs,
            validateur,
        )


# Valeur hors de l'intervalle numérique déclenche une erreur de validation

# Tempo

def test_tempo_invalide(validateur):
    """Vérifie qu'un tempo hors limites est refusé."""
    valeurs = valeurs_valides.copy()
    valeurs["tempo"] = 300.0

    with pytest.raises(ErreurValidation):
        valider_entrees(
            "Ma chanson",
            "Mon artiste",
            "pop",
            minutes_valides,
            secondes_valides,
            valeurs,
            validateur,
        )


# Volume

def test_volume_invalide(validateur):
    """Vérifie qu'un volume hors limites est refusé."""
    valeurs = valeurs_valides.copy()
    valeurs["loudness"] = 5.0

    with pytest.raises(ErreurValidation):
        valider_entrees(
            "Ma chanson",
            "Mon artiste",
            "pop",
            minutes_valides,
            secondes_valides,
            valeurs,
            validateur,
        )


# Durée

def test_duree_invalide(validateur):
    """Vérifie qu'une durée hors limites est refusée."""
    valeurs = valeurs_valides.copy()
    valeurs["duration_ms"] = 10000000

    with pytest.raises(ErreurValidation):
        valider_entrees(
            "Ma chanson",
            "Mon artiste",
            "pop",
            minutes_valides,
            secondes_valides,
            valeurs,
            validateur,
        )


# Valeur de tempo non numérique déclenche une erreur de validation

def test_valeur_non_numerique(validateur):
    """Vérifie qu'une valeur non numérique est refusée."""
    valeurs = valeurs_valides.copy()
    valeurs["tempo"] = "rapide"

    with pytest.raises(ErreurValidation):
        valider_entrees(
            "Ma chanson",
            "Mon artiste",
            "pop",
            minutes_valides,
            secondes_valides,
            valeurs,
            validateur,
        )


# Durée négative déclenche une erreur de validation

def test_minutes_negatives(validateur):
    """Vérifie que les minutes négatives sont refusées."""
    with pytest.raises(ErreurValidation):
        valider_entrees(
            "Ma chanson",
            "Mon artiste",
            "pop",
            -1,
            0,
            valeurs_valides,
            validateur,
        )


# Secondes > 59 déclenchent une erreur de validation

def test_secondes_invalides(validateur):
    """Vérifie que 60 secondes sont refusées."""
    with pytest.raises(ErreurValidation):
        valider_entrees(
            "Ma chanson",
            "Mon artiste",
            "pop",
            3,
            60,
            valeurs_valides,
            validateur,
        )


# ------ Tests directs de VariableNumerique ------


# Vérification qu'une borne minimale d'un intervalle est acceptée en créant une variable

def test_borne_minimale_acceptee():
    """Vérifie que la borne minimale est acceptée."""
    variable = VariableNumerique("danceability", 0, 1)
    variable.valider(0)


# Vérification qu'une borne maximale d'un intervalle est acceptée en créant une variable

def test_borne_maximale_acceptee():
    """Vérifie que la borne maximale est acceptée."""
    variable = VariableNumerique("danceability", 0, 1)
    variable.valider(1)


# ---- Valeur sous le minimum et au-dessus du maximum refusées ----

# Dansabilité

@pytest.mark.parametrize("valeur", [-0.1, 1.1])
def test_dansabilite_hors_limites(valeur):
    """Vérifie que les valeurs hors limites sont refusées."""
    variable = VariableNumerique("danceability", 0, 1)

    with pytest.raises(ErreurValidation):
        variable.valider(valeur)


# Énergie

@pytest.mark.parametrize("valeur", [-0.1, 1.1])
def test_energie_hors_limites(valeur):
    """Vérifie que les valeurs hors limites sont refusées."""
    variable = VariableNumerique("energy", 0, 1)

    with pytest.raises(ErreurValidation):
        variable.valider(valeur)
