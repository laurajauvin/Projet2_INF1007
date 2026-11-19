import json
import joblib
import numpy as np
import pandas as pd
import pytest
from src.donnees import charger_donnees
from src.entrainement import (calculer_importances, calculer_mae, calculer_reference, creer_metadonnees, creer_modele, entrainer_modele, predire, sauvegarder_resultats)
from validation import valider_entrees, ErreurValidation

#Colonnes que le modèle utilise comme entrée
COLONNES_ENTREE = [
    "danceability",
    "energy",
    "tempo",
    "loudness",
    "duration_ms",
    "track_genre",
]

df = charger_donnees("dataset_30000_clean.csv")

#Fixture: charge le dataset nettoyé et en choisit un petit échantillon pour éviter que les tests ne prennent trop de temps
@pytest.fixture(scope = "module") #Avec scope="module", les 60 lignes sont préparées une fois, puis réutilisées pour les autres tests de test_modele.py (évite de tout recréer l'échantillon à chaque fois)
def df_test() -> pd.DataFrame:
    """Charge le CSV réel et en prend 60 lignes pour les tests."""
    df = charger_donnees("dataset_30000_clean.csv")
    return df.sample(n=60, random_state=42)


@pytest.fixture(scope="module")
def modele_entraine(df_test: pd.DataFrame):
    """Entraîne le pipeline sur l'échantillon réel utilisé dans ces tests."""
    return entrainer_modele(
        creer_modele(),
        df_test[COLONNES_ENTREE],
        df_test["popularity"],
    )


def test_reference_calcule_mae() -> None:
    """Vérifie la MAE de référence sur un exemple calculable à la main."""
    y_train = pd.Series([10, 20])
    y_test = pd.Series([10, 30])

    resultat = calculer_reference(y_train, y_test)

    assert resultat == pytest.approx(10.0)


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


def test_modele_contient_preprocesseur_et_foret() -> None:
    """Vérifie que le pipeline contient les deux étapes attendues."""
    modele = creer_modele()

    assert "columntransformer" in modele.named_steps
    assert "randomforestregressor" in modele.named_steps


def test_modele_entraine_produit_une_prediction_par_chanson(
    df_test: pd.DataFrame,
    modele_entraine,
) -> None:
    """Vérifie que le modèle entraîné prédit chaque ligne du vrai échantillon."""
    predictions = predire(modele_entraine, df_test[COLONNES_ENTREE])

    assert len(predictions) == len(df_test)
    assert np.all(predictions >= 0)
    assert np.all(predictions <= 100)


def test_importances_sont_regroupees(
    modele_entraine,
) -> None:
    """Vérifie les noms des caractéristiques et la somme des importances."""
    importances = calculer_importances(modele_entraine)

    assert "Genre musical" in importances
    assert sum(importances.values()) == pytest.approx(1.0)


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
    assert metadonnees["intervalles_numeriques"]["energy"]["min"] == pytest.approx(
        df_test["energy"].min()
    )


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


# ------ Tests de validation sur les entrées de l'utilisateur ------

# Données valides pour les tests
valeurs_valides = {
    "danceability": 0.5,
    "energy": 0.8,
    "tempo": 120.0,
    "loudness": -10.0,
    "duration_ms": 180000
}


def test_entrees_valides():
    valider_entrees(
        "Ma chanson", "Mon artiste", "pop", valeurs_valides
    )


def test_titre_vide():
    with pytest.raises(ErreurValidation):
        valider_entrees("", "Mon artiste", "pop", valeurs_valides)


def test_artiste_vide():
    with pytest.raises(ErreurValidation):
        valider_entrees("Ma chanson", "", "pop", valeurs_valides)


def test_genre_invalide():
    with pytest.raises(ErreurValidation):
        valider_entrees(
            "Ma chanson", "Mon artiste", "jazz-invalide", valeurs_valides
        )


def test_dansabilite_invalide():
    valeurs = valeurs_valides.copy()
    valeurs["danceability"] = 1.5

    with pytest.raises(ErreurValidation):
        valider_entrees("Ma chanson", "Mon artiste", "pop", valeurs)


def test_energie_invalide():
    valeurs = valeurs_valides.copy()
    valeurs["energy"] = -0.5

    with pytest.raises(ErreurValidation):
        valider_entrees("Ma chanson", "Mon artiste", "pop", valeurs)


def test_tempo_invalide():
    valeurs = valeurs_valides.copy()
    valeurs["tempo"] = 300.0

    with pytest.raises(ErreurValidation):
        valider_entrees("Ma chanson", "Mon artiste", "pop", valeurs)


def test_volume_invalide():
    valeurs = valeurs_valides.copy()
    valeurs["loudness"] = 5.0

    with pytest.raises(ErreurValidation):
        valider_entrees("Ma chanson", "Mon artiste", "pop", valeurs)


def test_duree_invalide():
    valeurs = valeurs_valides.copy()
    valeurs["duration_ms"] = 10000000

    with pytest.raises(ErreurValidation):
        valider_entrees("Ma chanson", "Mon artiste", "pop", valeurs)


def test_valeur_non_numerique():
    valeurs = valeurs_valides.copy()
    valeurs["tempo"] = "rapide"

    with pytest.raises(ErreurValidation):
        valider_entrees("Ma chanson", "Mon artiste", "pop", valeurs)
