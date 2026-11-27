# Décisions du projet 2 — INF1007

Les validations A, B, C et D ont été présentées à notre professeure le vendredi 13 novembre 2026, de 12 h 45 à 14 h 45.

| Choix | Options envisagées | Choix retenu et justification | Validation |
|---|---|---|---|
| A — Domaine | Musique, cinéma, sport | Nous avons choisi la musique, car elle permet d'explorer les liens entre les caractéristiques musicales et la popularité. L'utilisateur peut entrer les caractéristiques d'une chanson pour obtenir une prédiction de popularité. | Validé le 13 novembre 2026 |
| B — Jeu de données | Différents jeux de données disponibles | Nous avons choisi le Spotify Tracks Dataset de Kaggle. Il contient environ 114 000 observations et 21 colonnes, avec des variables numériques et catégorielles ainsi qu'une cible numérique. Nous avons vérifié le nombre de lignes, les types de colonnes et la licence. | Validé le 13 novembre 2026 |
| C — Variable cible | Différentes variables disponibles | Nous avons choisi popularity, un score numérique entre 0 et 100, comme variable cible. | Validé le 13 novembre 2026 |
| C — Métrique | MAE, RMSE | Nous avons choisi la MAE, car elle mesure l'écart absolu moyen entre les prédictions et les valeurs réelles, en points de popularité. | Validé le 13 novembre 2026 |
| C — Modèle | Régression linéaire, arbre de décision, forêt aléatoire | Nous avons choisi Random Forest Regressor, qui combine plusieurs arbres de décision et peut apprendre des relations complexes entre les variables. | Validé le 13 novembre 2026 |
| D — Interface | Tkinter, PyQt, PySide, Streamlit, Gradio | Nous avons choisi Streamlit pour créer une interface interactive en Python. Une maquette papier de la fenêtre principale a été présentée. | Validé le 13 novembre 2026 |
| Prétraitement | Encodage manuel ou OneHotEncoder | Nous avons utilisé OneHotEncoder pour transformer le genre musical en colonnes numériques. | Décision de l'équipe |
| Séparation des données | Différentes proportions | Nous avons choisi 80 % pour l'entraînement et 20 % pour le test, avec random_state=42. | Décision de l'équipe |
| Référence | Prédiction constante ou modèle simple | Nous avons choisi la moyenne des popularités de l'ensemble d'entraînement comme référence de comparaison. | Décision de l'équipe |
| Sauvegarde | Réentraînement ou sauvegarde du modèle | Nous avons choisi Joblib pour sauvegarder le modèle et JSON pour conserver les métadonnées. | Décision de l'équipe |