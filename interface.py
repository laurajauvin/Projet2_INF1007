import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import joblib

if "historique" not in st.session_state:
    st.session_state.historique = []

modele = joblib.load("modeles/modele_popularite.joblib")

df = pd.read_csv("dataset_30000_clean.csv")

# Configuration de la page
st.set_page_config(
    page_title="Prédiction de popularité musicale",
    page_icon="🎵",
    layout="wide"
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

# COLONNE DE GAUCHE (Caractéristiques musicales)
with col_caracteristique:
    st.subheader("Caractéristiques musicales")
    st.markdown("<div style='padding-top: 10px;'></div>",
        unsafe_allow_html=True)

    
    sous_col_gauche, sous_col_droite = st.columns([1, 1])
    
    st.markdown("""
        <style>

        div[data-testid="stTextInput"] label p{
            font-size: 23px !important;
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
            font-size: 23px !important;
            color: var(--text-color) ;
        }

        div[data-testid="stSelectbox"] {
            max-width: 70%;
            width: 100%;
            padding-bottom: 20px;
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
            font-size: 23px !important;
            color: var(--text-color) ;
        }

        div[data-testid="stNumberInput"] {
            max-width: 70%;
            width: 100%;
            padding-bottom: 20px;
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

        </style>
    """, unsafe_allow_html=True)

    with sous_col_gauche:

        chanson = st.text_input("Chanson")

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

        artiste = st.text_input("Artiste")

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


#Prédictions

    st.markdown("<div style='padding-top: 15px;'></div>",
        unsafe_allow_html=True)

    st.subheader("Prédiction")
    

    if st.button("Prédire la popularité"):

        if genre is None:
            st.warning("Veuillez sélectionner un genre musical.")

        else:
            donnees = pd.DataFrame([{
                "danceability": dansabilite,
                "duration_ms": duree,
                "tempo": tempo,
                "energy": energie,
                "loudness": volume,
                "track_genre": genre
            }])

            prediction = modele.predict(donnees)[0]

            st.session_state.historique.append({
                "chanson": chanson,
                "artiste": artiste,
                "prediction": round(prediction, 1)
            })
                
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
    fig, ax = plt.subplots(figsize=(7, 5.5))

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
    ax.set_xlim(0, max(df_importances["Importance"]) * 1.15)

    fig.tight_layout()

    st.pyplot(fig, use_container_width=True)

    plt.close(fig)
    
#Historique des predictions

    st.markdown("<div style='padding-top: 40px;'></div>",
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