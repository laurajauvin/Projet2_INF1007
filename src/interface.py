import joblib
import math
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import json
import os
import pickle
import zlib
from src.entrainement import predire_popularite, calculer_importances
from src.exceptions import ErreurValidation
from src.validation import valider_entrees, creer_validateur



def afficher_interface() -> None:
    """Construit et affiche l'interface de streamlit
       Ne retourne aucune valeur"""


    # ---- Configurer l'apparence de la page streamlit ----

    st.set_page_config(
        page_title="Prédiction de popularité musicale",
        page_icon="🎵",
        layout="wide"  # Mise en page large
    )


    # ---- Construire les chemins des fichiers ----
    
    dossier_src = os.path.dirname(os.path.abspath(__file__)) # Trouver le dossier src
    dossier_projet = os.path.dirname(dossier_src) # Remonter au dossier principal du projet

    chemin_metadonnees = os.path.join(
        dossier_projet, "modeles", "metadonnees.json" # Relie les fichiers entre eux
    )

    chemin_modele = os.path.join(
        dossier_projet, "modeles", "modele_popularite.joblib"
    )

    chemin_historique = os.path.join(
        dossier_projet, "historique.csv"
    )


    # ---- Vérifier que les fichiers nécessaires existent ----

    if not os.path.isfile(chemin_metadonnees):
        st.error("Le fichier des métadonnées est introuvable.") # Si manque un fichier -> erreur affichée et arrêt de l'application
        st.stop()

    if not os.path.isfile(chemin_modele):
        st.error("Le fichier du modèle entraîné est introuvable.")
        st.stop()


    # ---- Charger les métadonnées ----

    try:
        with open(chemin_metadonnees, "r", encoding="utf-8") as fichier:  # Ouvre en lecture et ferme automatiquement
            metadonnees = json.load(fichier)

    except (json.JSONDecodeError, OSError, UnicodeDecodeError):
        st.error(
            "Le fichier des métadonnées est invalide ou impossible à lire."
        )
        st.stop()

    #Crée le validateur en lisant les genres et bornes dans le fichier json
    validateur = creer_validateur()


    # ---- Récupéree les intervalles numériques définis dans le JSON ----

    intervalles = metadonnees["intervalles_numeriques"]

    dansabilite_min = intervalles["danceability"]["min"]
    dansabilite_max = intervalles["danceability"]["max"] 

    energie_min = intervalles["energy"]["min"]
    energie_max = intervalles["energy"]["max"]

    tempo_min = intervalles["tempo"]["min"]
    tempo_max = intervalles["tempo"]["max"]

    volume_min = intervalles["loudness"]["min"]
    volume_max = intervalles["loudness"]["max"]

    # Conversion de la durée en millisecondes pour qu'elle soit en secondes
    duree_min = math.ceil(intervalles["duration_ms"]["min"] / 1000)
    duree_max = math.floor(intervalles["duration_ms"]["max"] / 1000)   

    # Récupérer la liste des genres acceptés
    genres = metadonnees["genres_possibles"]


    # ---- Charger le modèle Random Forest déjà entraîné et sauvegardé ----

    try:
        modele = joblib.load(chemin_modele)

    except (
        OSError,
        EOFError,
        ValueError,
        TypeError,
        KeyError,
        AttributeError,
        ImportError,
        IndexError,
        pickle.UnpicklingError,
        zlib.error
    ):
        st.error(
            "Le fichier du modèle est invalide ou impossible à charger."
        )
        st.stop()


    # ---- Initialiser l'historique dans la session Streamlit s'il n'existe pas ----

    if "historique" not in st.session_state:
        st.session_state.historique = []

        # Vérifier si un ancien fichier historique existe
        if os.path.isfile(chemin_historique):

            try:
                historique_df = pd.read_csv(chemin_historique)

                # Défini les colonnes indispensables pour l'historique
                colonnes_requises = [
                    "chanson",
                    "artiste",
                    "prediction"
                ]

                # Vérifier que toutes les colonnes nécessaires sont présentes
                if all(colonne in historique_df.columns
                    for colonne in colonnes_requises):

                    # Convertir les données en liste de dictionnaires
                    st.session_state.historique = historique_df[
                        colonnes_requises
                    ].to_dict("records")

                else:
                    st.warning(
                        "Le fichier historique.csv ne possède pas "
                        "les colonnes attendues. Un nouvel historique sera utilisé."
                    )

            # Gérer les erreurs possibles lors de la lecture du CSV
            except pd.errors.EmptyDataError:
                st.warning(
                    "Le fichier historique.csv est vide. "
                    "Un nouvel historique sera utilisé."  # Fichier Vide
                )

            except pd.errors.ParserError:
                st.warning(
                    "Le fichier historique.csv est mal formaté. "
                    "Un nouvel historique sera utilisé."  # Mauvais format
                )

            except UnicodeDecodeError:
                st.warning(
                    "Le fichier historique.csv ne peut pas être lu "
                    "à cause de son encodage."  # Problème d'encodage
                )


    # ---- Marges générales de l'application ----

    st.markdown("""
    <style>
        [data-testid="stMainBlockContainer"] {
            padding-left: 9rem !important;
            padding-right: 9rem !important;
        }
    </style>
    """, unsafe_allow_html=True) # unsafe_allow_html=True -> Autorise l'interprétation du HTML
                                 # Utilisation des langages html et CSS pour l'apparence de l'interface
                                 # (espace, padding, couleurs, police d'écriture, etc)
                                 # st.markdown() -> Affiche du texte formaté
                                 # <style> -> Permet de modifier le style


    # ---- Taille du texte ----

    st.markdown("""
        <style>

        /* Titre principal : st.title() */
        h1 {
            font-size: 60px !important;
        }

        /* Sous-titres : st.header() */
        h2 {
            font-size: 35px !important;
        }

        /* Sous-sous-titres : st.subheader() */
        h3 {
            font-size: 35px !important;
        }

        div[data-testid="stText"] {
            font-size: 20px !important;
        }

        div[data-testid="stText"] pre {
            font-size: 20px !important;
        }

        div[data-testid="stMarkdownContainer"] p {
            font-size: 20px !important;
        }

        </style>
        """, unsafe_allow_html=True)


    # ---- Titre Affiché ----

    st.markdown(
        "<h1 style = 'text-align: center; padding-bottom: 50px;'>🎶 Prédiction de popularité musicale 🎶</h1>",
        unsafe_allow_html=True
    )

    #text-align: center -> Centre le titre


    # ---- Diviser l'interface en deux colonnes ----

    col_caracteristique, col_importance = st.columns([1, 1]) # Chaque colonne occupe la moitié de l'écran


    # ---- # COLONNE DE GAUCHE (Entrées des caractéristiques) ----

    with col_caracteristique:
        st.subheader("Caractéristiques musicales")
        st.markdown("<div style='padding-top: 10px;'></div>",
            unsafe_allow_html=True)


        # ---- Les caractéristiques séparées en deux sous-colonnes ----

        sous_col_gauche, sous_col_droite = st.columns([1, 1])


        # ---- Apparence des entrées et étiquettes modifiée en CSS ----

        st.markdown("""
            <style>

            div[data-testid="stTextInput"] label p{
                font-size: 22px !important;
                color: var(--text-color) ;
            }

            div[data-testid="stTextInput"] {
                max-width: 70%;
                width: 100%;
                padding-bottom: 20px;
            }   
            
            div[data-testid="stTextInput"] input {
                width: 100%;
                box-sizing: border-box;
                background-color: var(--secondary-background-color);
                color: var(--text-color);
                border-radius: 20px;
                border: none;
                font-size: 17px
            }


            div[data-testid="stSelectbox"] label p{
                font-size: 22px !important;
                color: var(--text-color) ;
            }

            div[data-testid="stSelectbox"] {
                max-width: 70%;
                width: 100%;
                padding-bottom: 46px;
            }   

            div[data-testid="stSelectbox"] input {
                width: 100%;
                box-sizing: border-box;
                background-color: var(--secondary-background-color);
                color: var(--text-color);
                border-radius: 20px;
                border: none;
                font-size: 17px
            }


            div[data-testid="stNumberInput"] label p{
                font-size: 22px !important;
                color: var(--text-color) ;
            }

            div[data-testid="stNumberInput"] {
                max-width: 70%;
                width: 100%;
                padding-bottom: 0px
            }   
            
            div[data-testid="stNumberInput"] input {
                width: 100%;
                box-sizing: border-box;
                background-color: var(--secondary-background-color);
                color: var(--text-color);
                border-radius: 20px;
                border: none;
                font-size: 17px
            }

            [data-testid="stCaptionContainer"] {
                margin-bottom: 0.4rem;
            }

            </style>
        """, unsafe_allow_html=True)


        # ---- Sous-colonne gauche des caractéristiques ----

        with sous_col_gauche:


            # Chanson
            chanson = st.text_input("Chanson")  # Entrer le nom de la chanson


            # Genre
            genre = st.selectbox(   # Sélectionner un genre de musique
                "Genre",
                genres,
                index = None,
                placeholder="Rechercher un genre..."
            )


            with st.container(gap="xsmall"):  # Réduit l'espace entre les entrées


                # Dansabilité
                dansabilite = st.number_input(   # Nombre entre 0 et 1 selon si on peut danser ou non sur la chanson
                    "Dansabilité",
                    step = 0.01,
                    format = "%.2f"
                )

                st.caption(
                    f"Dansabilité acceptée : {dansabilite_min:.2f} à {dansabilite_max:.2f}"
                )  # Indique à l'utilisateur l'intervalle de valeurs acceptées


            # Durée
                col_minutes, col_secondes = st.columns(2, gap="xxsmall")

                with col_minutes:      # Entrée pour les minutes
                    minutes = st.number_input(
                        "Minutes",
                        step=1
                    )

                with col_secondes:      # Entrée pour les secondes
                    secondes = st.number_input(
                        "Secondes",
                        step=1
                    )

                duree = int((minutes * 60 + secondes) * 1000)

                st.caption(
                    f"Durée acceptée : de "
                    f"{duree_min // 60} min {duree_min % 60} s "
                    f"à {duree_max // 60} min {duree_max % 60} s"
                )


        # ---- Sous-colonne droite des caractéristiques ----   

        with sous_col_droite:


            # Artiste
            artiste = st.text_input("Artiste")


            with st.container(gap="xsmall"):


                # Tempo
                tempo = st.number_input(
                    "Tempo (bpm)",
                    step = 1.0,
                    format = "%.2f"
                )

                st.caption(
                    f"Tempo accepté : {tempo_min:.2f} à {tempo_max:.2f}"
                )


                # Energie
                energie = st.number_input(
                    "Énergie",
                    step=0.01,
                    format = "%.2f"
                )

                st.caption(
                    f"Énergie acceptée : {energie_min:.2f} à {energie_max:.2f}"
                )


                # Volume
                volume = st.number_input(
                    "Volume (dB)",
                    step=0.1,
                    format="%.1f"
                )

                volume_min_affiche = math.ceil(volume_min * 10) / 10
                volume_max_affiche = math.floor(volume_max * 10) / 10

                st.caption(
                    f"Volume accepté : {volume_min_affiche:.1f} à {volume_max_affiche:.1f}"
                )


        # ---- Prédictions ----

        st.markdown("<div style='padding-top: 15px;'></div>",
            unsafe_allow_html=True)   # Ajout d'espace

        st.subheader("Prédiction")  # Sous-titre
        

        if st.button("Prédire la popularité"):      # Lorsque l'on appuie sur le bouton "Prédire la popularité"
            valeurs = {
                "danceability": dansabilite,
                "energy": energie,
                "tempo": tempo,
                "loudness": volume,
                "duration_ms": duree
            }

            # Vérifie si toutes les entrées sont valides
            try:
                valider_entrees(chanson, artiste, genre, minutes, secondes, valeurs, validateur)   

            except ErreurValidation as erreur:
                st.error(str(erreur))

            # Si oui, on lance la prédiction
            else:
                donnees = pd.DataFrame([{
                    "danceability": dansabilite,
                    "duration_ms": duree,
                    "tempo": tempo,
                    "energy": energie,
                    "loudness": volume,
                    "track_genre": genre
                }])

                prediction = predire_popularite(modele, donnees)  # Fonction de "entraînement" -> retourne la popularité prédite

                # Créer une nouvelle entrée
                nouveau_resultat = {
                    "chanson": chanson,
                    "artiste": artiste,
                    "prediction": np.round(prediction, 1)
                }

                # Ajouter à l'historique en mémoire
                st.session_state.historique.append(nouveau_resultat)

                # Sauvegarder tout l'historique dans le CSV
                historique_df = pd.DataFrame(st.session_state.historique)

                historique_df.to_csv(
                    chemin_historique,
                    index=False,
                    encoding="utf-8-sig"
                )

                # Affiche la popularité prédite
                st.markdown(
                    f"""
                    <div style="
                        background-color: #0e2f24;
                        color: #7ed5b0;
                        padding: 15px;
                        border-radius: 10px;
                        font-size: 20px;
                        width: 70%;
                    ">
                        Popularité prédite : {prediction:.1f}/100
                    </div>
                    """,
                    unsafe_allow_html=True
                )


    # ---- COLONNE DE DROITE (Importance des caractéristiques) ----

    with col_importance:
        st.subheader("Importance des caractéristiques")  # Sous-titre

        st.markdown("<div style='padding-top: 20px;'></div>",
            unsafe_allow_html=True)

        # Récupérer les importances regroupées par caractéristique (fonction de "entrainement")
        importances_caracteristiques = calculer_importances(modele) # Stockées dans un dictionnaire

        # Transforme le dictionnaire en DataFrame
        df_importances = pd.DataFrame({
            "Caractéristique": importances_caracteristiques.keys(), 
            "Importance": importances_caracteristiques.values()
        })

        # Trier de la plus faible à la plus élevée
        df_importances = df_importances.sort_values(
        by="Importance",
        key=lambda valeurs: valeurs,
        ascending=True)


        # ---- Créer le graphique ----

        fig, ax = plt.subplots(figsize=(7, 6.25))

        ax.barh(  # Diagramme à barre
            df_importances["Caractéristique"],
            df_importances["Importance"],
            color= "#665388",
            label="Importance"
        )

        ax.legend()

        # Parcours la liste d'importances pour les afficher sur le diagramme
        for i, valeur in enumerate(df_importances["Importance"]):
            ax.text(
                valeur,
                i,
                f" {valeur:.3f}",
                va="center")

        # Titres (axes)
        ax.set_xlabel("Importance relative")
        ax.set_title("Importance des caractéristiques")
        ax.set_ylabel("Caractéristiques musicales")
        ax.tick_params(
            axis="both",
            labelsize=11
        )

        # Ajouter une grille pour faciliter la lecture
        ax.grid(
            axis="x",
            linestyle="--",
            alpha=0.4
        )

        # Ajuster les limites de l'axe horizontal
        ax.set_axisbelow(True)
        ax.set_xlim(0, max(df_importances["Importance"]) * 1.15)

        # Ajuster les espacements
        fig.tight_layout()

        # Afficher le graphique dans Streamlit
        st.pyplot(fig, use_container_width=True)

        # Fermer la figure
        plt.close(fig)

        
    # ---- Historique des predictions ----

        st.markdown("<div style='padding-top: 58px;'></div>",
            unsafe_allow_html=True)

        st.subheader("Historique des prédictions")

        
        # ---- L'historique en deux sous-colonnes ----

        col_chanson, col_score = st.columns([2, 1])   # Colonne des chansons et artistes occupe plus d'espace

        # Colonne des chansons
        with col_chanson:
            st.markdown("#### Chanson")

        # Colonne des scores
        with col_score:
            st.markdown("#### Score")

        # Parcourt toutes les prédictions enregistrées dans l'historique
        for resultat in st.session_state.historique:

            # Pour chaque prédiction, créer deux colonnes et afficher la chanson, l'artiste et le score
            col_chanson, col_score = st.columns([2, 1])

            with col_chanson:
                st.write(f'{resultat["chanson"]} ~ {resultat["artiste"]}')

            with col_score:
                st.write(f'{resultat["prediction"]}/100')