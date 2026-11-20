import joblib
import math
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import json
import os
from .entrainement import predire_popularite
from .exceptions import ErreurValidation
from .validation import valider_entrees



def afficher_interface() -> None:
    """Construit et affiche l'interface de streamlit"""

    # Configuration de la page
    st.set_page_config(
        page_title="Prédiction de popularité musicale",
        page_icon="🎵",
        layout="wide"
    )

    # Construire les chemins des fichiers
    dossier_src = os.path.dirname(os.path.abspath(__file__))
    dossier_projet = os.path.dirname(dossier_src)

    chemin_metadonnees = os.path.join(
        dossier_projet, "modeles", "metadonnees.json"
    )

    chemin_modele = os.path.join(
        dossier_projet, "modeles", "modele_popularite.joblib"
    )

    chemin_historique = os.path.join(
        dossier_projet, "historique.csv"
    )


    # Vérifier que les fichiers nécessaires existent
    if not os.path.exists(chemin_metadonnees):
        st.error("Le fichier des métadonnées est introuvable.")
        st.stop()

    if not os.path.exists(chemin_modele):
        st.error("Le fichier du modèle entraîné est introuvable.")
        st.stop()


    # Charger les métadonnées
    try:
        with open(chemin_metadonnees, "r", encoding="utf-8") as fichier:
            metadonnees = json.load(fichier)

    except (json.JSONDecodeError, OSError, UnicodeDecodeError):
        st.error(
            "Le fichier des métadonnées est invalide ou impossible à lire."
        )
        st.stop()



    # Attitrer chaque contrainte à une variable
    intervalles = metadonnees["intervalles_numeriques"]

    dansabilite_min = intervalles["danceability"]["min"]
    dansabilite_max = intervalles["danceability"]["max"]

    energie_min = intervalles["energy"]["min"]
    energie_max = intervalles["energy"]["max"]

    tempo_min = intervalles["tempo"]["min"]
    tempo_max = intervalles["tempo"]["max"]

    volume_min = intervalles["loudness"]["min"]
    volume_max = intervalles["loudness"]["max"]

    duree_min = math.ceil(intervalles["duration_ms"]["min"] / 1000)
    duree_max = math.floor(intervalles["duration_ms"]["max"] / 1000)   

    genres = metadonnees["genres_possibles"]

    # Charger le modèle déjà entraîné
    try:
        modele = joblib.load(chemin_modele)

    except Exception:
        st.error(
            "Le fichier du modèle est invalide ou impossible à charger."
        )
        st.stop()

    # Charger l'historique existant au démarrage
    if "historique" not in st.session_state:

        st.session_state.historique = []

        if os.path.exists(chemin_historique):

            try:
                historique_df = pd.read_csv(chemin_historique)

                # Vérifier que les colonnes nécessaires existent
                colonnes_requises = [
                    "chanson",
                    "artiste",
                    "prediction"
                ]

                if all(colonne in historique_df.columns
                    for colonne in colonnes_requises):

                    st.session_state.historique = historique_df[
                        colonnes_requises
                    ].to_dict("records")

                else:
                    st.warning(
                        "Le fichier historique.csv ne possède pas "
                        "les colonnes attendues. Un nouvel historique sera utilisé."
                    )

            except pd.errors.EmptyDataError:
                st.warning(
                    "Le fichier historique.csv est vide. "
                    "Un nouvel historique sera utilisé."
                )

            except pd.errors.ParserError:
                st.warning(
                    "Le fichier historique.csv est mal formaté. "
                    "Un nouvel historique sera utilisé."
                )

            except UnicodeDecodeError:
                st.warning(
                    "Le fichier historique.csv ne peut pas être lu "
                    "à cause de son encodage."
                )

    # Marges générales de l'application
    st.markdown("""
    <style>
        [data-testid="stMainBlockContainer"] {
            padding-left: 9rem !important;
            padding-right: 9rem !important;
        }
    </style>
    """, unsafe_allow_html=True)


    #Taille du texte

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

    # Titre
    st.markdown(
        "<h1 style = 'text-align: center; padding-bottom: 50px;'>🎶 Prédiction de popularité musicale 🎶</h1>",
        unsafe_allow_html=True
    )

    #st.markdown() -> Affiche du texte formaté
    #<h1> -> Crée un grand titre HTML
    #style =	-> Permet de modifier le style
    #text-align: center -> Centre le titre
    #unsafe_allow_html=True -> Autorise l'interprétation du HTML


    #Diviser l'interface en deux colonnes

    col_caracteristique, col_importance = st.columns([1, 1])

    # Colonne gauche des caractéristiques

    with col_caracteristique:
        st.subheader("Caractéristiques musicales")
        st.markdown("<div style='padding-top: 10px;'></div>",
            unsafe_allow_html=True)

        
        sous_col_gauche, sous_col_droite = st.columns([1, 1])
        
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

        with sous_col_gauche:

            # Chanson
            chanson = st.text_input("Chanson")

            # Genre
            genre = st.selectbox(
                "Genre",
                genres,
                index = None,
                placeholder="Rechercher un genre..."
            )

            with st.container(gap="xsmall"):
                # Dansabilité
                dansabilite = st.number_input(
                    "Dansabilité",
                    step = 0.01,
                    format = "%.2f"
                )

                st.caption(
                    f"Dansabilité acceptée : {dansabilite_min:.2f} à {dansabilite_max:.2f}"
                )

            # Durée
                col_minutes, col_secondes = st.columns(2, gap="xxsmall")

                with col_minutes:
                    minutes = st.number_input(
                        "Minutes",
                        step=1
                    )

                with col_secondes:
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


        # Colonne droite des caractéristiques    

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


        #Prédictions

        st.markdown("<div style='padding-top: 15px;'></div>",
            unsafe_allow_html=True)

        st.subheader("Prédiction")
        

        if st.button("Prédire la popularité"):
            valeurs = {
                "danceability": dansabilite,
                "energy": energie,
                "tempo": tempo,
                "loudness": volume,
                "duration_ms": duree
            }

            try:
                valider_entrees(chanson, artiste, genre, minutes, secondes, valeurs)

            except ErreurValidation as erreur:
                st.error(str(erreur))

            else:
                donnees = pd.DataFrame([{
                    "danceability": dansabilite,
                    "duration_ms": duree,
                    "tempo": tempo,
                    "energy": energie,
                    "loudness": volume,
                    "track_genre": genre
                }])

                prediction = predire_popularite(modele, donnees)

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


    # COLONNE DE DROITE (Importance des caractéristiques)
    with col_importance:
        st.subheader("Importance des caractéristiques")

        st.markdown("<div style='padding-top: 20px;'></div>",
            unsafe_allow_html=True)

        # Récupérer le modèle et le préprocesseur
        random_forest = modele.named_steps["randomforestregressor"]
        preprocesseur = modele.named_steps["columntransformer"]

        importances = random_forest.feature_importances_
        noms = preprocesseur.get_feature_names_out()

        # Regrouper les importances par caractéristique
        importances_caracteristiques = {
            "Dansabilité": 0,
            "Énergie": 0,
            "Tempo": 0,
            "Volume": 0,
            "Durée": 0,
            "Genre musical": 0
        }

        for nom, importance in zip(noms, importances):

            if "genre__track_genre" in nom:
                importances_caracteristiques["Genre musical"] += importance

            elif "danceability" in nom:
                importances_caracteristiques["Dansabilité"] = importance

            elif "energy" in nom:
                importances_caracteristiques["Énergie"] = importance

            elif "tempo" in nom:
                importances_caracteristiques["Tempo"] = importance

            elif "loudness" in nom:
                importances_caracteristiques["Volume"] = importance

            elif "duration_ms" in nom:
                importances_caracteristiques["Durée"] = importance

        # Créer un DataFrame et trier les importances
        df_importances = pd.DataFrame({
            "Caractéristique": importances_caracteristiques.keys(),
            "Importance": importances_caracteristiques.values()
        })

        df_importances = df_importances.sort_values(
            by="Importance",
            ascending=True
        )

        # Créer le graphique
        fig, ax = plt.subplots(figsize=(7, 6.25))

        ax.barh(
            df_importances["Caractéristique"],
            df_importances["Importance"],
            color= "#665388"
        )

        for i, valeur in enumerate(df_importances["Importance"]):
            ax.text(
                valeur,
                i,
                f" {valeur:.3f}",
                va="center")

        ax.set_xlabel("Importance relative")
        ax.set_title("Importance des caractéristiques")
        ax.set_ylabel("Caractéristiques musicales")
        ax.tick_params(
            axis="both",
            labelsize=11
        )

        ax.grid(
            axis="x",
            linestyle="--",
            alpha=0.4
        )

        ax.set_axisbelow(True)
        ax.set_xlim(0, max(df_importances["Importance"]) * 1.15)

        fig.tight_layout()

        st.pyplot(fig, use_container_width=True)

        plt.close(fig)
        
    #Historique des predictions

        st.markdown("<div style='padding-top: 58px;'></div>",
            unsafe_allow_html=True)

        st.subheader("Historique des prédictions")

        col_chanson, col_score = st.columns([2, 1])

        with col_chanson:
            st.markdown("#### Chanson")
        with col_score:
            st.markdown("#### Score")

        for resultat in st.session_state.historique:
            col_chanson, col_score = st.columns([2, 1])

            with col_chanson:
                st.write(f'{resultat["chanson"]} - {resultat["artiste"]}')

            with col_score:
                st.write(f'{resultat["prediction"]}/100')