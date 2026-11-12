# Projet2_INF1007

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