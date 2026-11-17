import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# Configuration de la page
st.set_page_config(
    page_title="Prédiction de popularité musicale",
    page_icon="🎵",
    layout="wide"
)

# Titre
st.markdown(
    "<h1 style = 'text-align: center;'>🎶 Prédiction de popularité musicale 🎶</h1>",
    unsafe_allow_html=True
)

"""
st.markdown() -> Affiche du texte formaté
<h1> -> Crée un grand titre HTML
style =	-> Permet de modifier le style
text-align: center -> Centre le titre
unsafe_allow_html=True -> Autorise l'interprétation du HTML
"""

# Description
st.write(
    "Découvrez la popularité potentielle d'une chanson "
    "à partir de ses caractéristiques musicales."
)

#Diviser l'interface en deux colonnes

col_gauche, col_droite = st.columns([1, 1])

# COLONNE DE GAUCHE
with col_gauche:
    st.subheader("Caractéristiques musicales")
    st.badge("Enter", color="#E1B5FF")
    st.badge("Enter", color="#E1B5FF")
    st.badge("Enter", color="#E1B5FF")
    st.badge("Enter", color="#E1B5FF")
    st.badge("Enter", color="#E1B5FF")
    st.badge("Enter", color="#E1B5FF")

# COLONNE DE DROITE
with col_droite:
    st.subheader("Importance des caractéristiques")

