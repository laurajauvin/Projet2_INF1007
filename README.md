# Projet2_INF1007

---- Documentation officielle ----

Pandas : Getting started tutorials — documentation utilisée pour le chargement et l'exploration des données.
Scikit-learn : Getting Started — documentation utilisée pour construire, entraîner et évaluer le modèle.
Streamlit : Get started — documentation utilisée pour développer l'interface graphique.


---- Source des données ----

* Nom : Spotify Tracks Dataset
* Auteur : Maharshi Pandya
* Source : Kaggle – https://www.kaggle.com/datasets/maharshipandya/-spotify-tracks-dataset
* Jeu de données original : environ 114 000 chansons.
* Jeu de données utilisé : sélection aléatoire de 30 000 chansons.
* Licence : Open Data Commons Open Database License (ODbL) 1.0.

Le jeu de données a été sélectionné et réduit à 30 000 observations à l’aide d’un échantillonnage aléatoire (random_state=42).


---- Chargement et exploration des données ----

Nous avons utilisé la bibliothèque Pandas pour charger le jeu de données Spotify à partir d'un fichier CSV. Le jeu de données initial contenait 114 000 chansons et 21 colonnes. Nous avons ensuite sélectionné aléatoirement 30 000 chansons à l'aide de sample(), avec une valeur fixe de random_state=42 afin d'obtenir un échantillon reproductible.

Nous avons examiné les types de données avec info(), les statistiques descriptives avec describe() et les valeurs manquantes avec isnull().sum(). Nous avons également vérifié la présence de doublons et de valeurs suspectes, notamment les durées et les tempos égaux à zéro.

---- Nettoyage des données ----

L'exploration a révélé :

3 valeurs manquantes au total, dans les colonnes artists, album_name et track_name (une dans chaque colonne).

Aucun doublon dans notre échantillon.

1 chanson ayant une durée égale à zéro.

52 chansons ayant un tempo égal à zéro.

Aucune valeur hors de l'intervalle [0, 1] dans les variables musicales vérifiées.

Nous avons retiré la colonne Unnamed: 0, puisqu'elle correspond à un index inutile pour notre analyse. Nous avons également supprimé les chansons ayant une durée ou un tempo égal à zéro, car ces valeurs ne sont pas pertinentes pour notre modèle de prédiction.

Nous avons conservé les valeurs manquantes dans artists, album_name et track_name, puisque ces colonnes ne sont pas utilisées comme variables d'entrée du modèle. Elles peuvent toutefois être utiles pour identifier les chansons.

Après le nettoyage, notre jeu de données contient 29 947 chansons.

----Définition du problème----

Notre variable cible est la popularité d'une chanson (popularity). Comme nous cherchons une valeur numérique, il s'agit d'un problème de régression. Le modèle choisi est une forêt aléatoire de régression (RandomForestRegressor). Elle prédit une valeur numérique et peut apprendre des relations plus complexes entre les caractéristiques des chansons et leur popularité. Nous avons choisi le MAE comme métrique d'évaluation, car il mesure l'écart absolu moyen entre les valeurs de popularité prédites par notre modèle et les valeurs réelles. Cette métrique est facile à interpréter puisque la popularité est mesurée sur une échelle de 0 à 100. Par exemple, un MAE de 8 signifie que les prédictions s'écartent en moyenne de 8 points des valeurs réelles.


---- Séparation des données ----

Après avoir nettoyé notre dataset, nous avons séparé les 29 947 chansons en deux ensembles : 80 % pour l'entraînement et 20 % pour les tests. Les données d'entraînement permettent au modèle d'apprendre les relations entre les caractéristiques musicales et la popularité des chansons. Les données de test servent ensuite à évaluer sa capacité à prédire la popularité de chansons qu'il n'a pas utilisées pendant son apprentissage. Nous avons utilisé random_state=42 afin de rendre notre séparation reproductible.

---- Entraînement et évaluation ----

Sur l’ensemble de test, la forêt aléatoire obtient une MAE de 12,98, contre 19,01 pour la référence basée sur la moyenne. Elle réduit donc l’erreur moyenne d’environ 6 points de popularité et fait mieux que cette référence. Le graphique prédit contre réel montre que le modèle suit partiellement la tendance, mais que les prédictions restent dispersées autour de la diagonale. Elles sont concentrées surtout entre 10 et 60 : le modèle surestime souvent les chansons dont la popularité réelle est très basse et sous-estime les plus populaires. La MAE indique une erreur absolue moyenne d’environ 13 points. Les différentes caractéristiques audio et le genre aident le modèle dans sa prédiction, puisqu’il fait mieux que la référence, mais elles ne suffisent pas à prédire précisément la popularité de chaque chanson. 

---- Utilisation des décorateurs ----

Plusieurs décorateurs sont utilisés dans le projet :

- `@abstractmethod` définit la méthode `valider()` comme abstraite dans la classe parente `Variable`. Les classes enfants doivent donc implémenter leur propre validation.
- `@property` permet d'accéder au nom d'une variable avec `variable.nom`, tout en conservant une méthode qui contrôle cet accès.
- `@pytest.fixture` permet de préparer des données et des objets réutilisables dans plusieurs tests, évitant de répéter leur création.
- `@chronometrer` est un décorateur personnalisé qui mesure et affiche le temps d'exécution de la fonction `entrainer_modele()`, ce qui permet de connaître la durée nécessaire à l'entraînement du modèle.

---- Couverture des tests ----

Les tests sont exécutés avec `pytest` et la couverture du code est mesurée avec `pytest-cov`.

La dernière exécution a donné 29 tests réussis sur 29 et une couverture globale de 40 %.

