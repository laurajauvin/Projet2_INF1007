import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# Configuration de la page
st.set_page_config(
    page_title="Prédiction de popularité musicale",
    page_icon="🎵",
    layout="wide"
)

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

col_gauche, col_droite = st.columns([1, 1])

# COLONNE DE GAUCHE
with col_gauche:
    st.subheader("Caractéristiques musicales")
    
    sous_col_gauche, sous_col_droite = st.columns([1, 1])
    
    st.markdown("""
        <style>
        div[data-testid="stSelectbox"] label p{
            font-size: 23px !important;
            color: var(--text-color) ;
        }

        div[data-testid="stSelectbox"] {
            max-width: 70%;
            width: 100%;
        }   

        div[data-testid="stSelectbox"] input {
            width: 100%;
            box-sizing: border-box;
            background-color: var(--secondary-background-color);
            color: var(--text-color);
            border-radius: 20px;
            border: none;
            font-size: 18px
        }


        div[data-testid="stNumberInput"] label p{
            font-size: 23px !important;
            color: var(--text-color) ;
        }

        div[data-testid="stNumberInput"] {
            max-width: 70%;
            width: 100%;
        }   
        
        div[data-testid="stNumberInput"] input {
            width: 100%;
            box-sizing: border-box;
            background-color: var(--secondary-background-color);
            color: var(--text-color);
            border-radius: 20px;
            border: none;
            font-size: 18px
        }
        </style>
    """, unsafe_allow_html=True)

    with sous_col_gauche:

        df = pd.read_csv("dataset_30000_clean.csv")
        genres = sorted(df["track_genre"].unique())

        genre = st.selectbox(
            "Genre",
            genres,
            index=None,
            placeholder="Rechercher un genre..."
)

        dansabilite = st.number_input(
            "Dansabilité",
            min_value=0.0,
            max_value=1.0,
            step=0.01
        )

        duree = st.number_input(
            "Durée (ms)",
            min_value=1,
            step=1000
        )

    with sous_col_droite:
        tempo = st.number_input(
            "Tempo (bpm)",
            min_value=1.0,
            step=1.0
        )

        energie = st.number_input(
            "Énergie",
            min_value=0.0,
            max_value=1.0,
            step=0.01
        )

        volume = st.number_input(
            "Volume (dB)",
            step=0.1
        )


# COLONNE DE DROITE
with col_droite:
    st.subheader("Importance des caractéristiques")
