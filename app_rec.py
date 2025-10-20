import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


#  Configuration et style personnalisé

st.set_page_config(page_title=" Recommandation de Films", layout="wide")

# CSS custom
st.markdown("""
    <style>
    /* Global style */
    body {
        background-color: #4;
        color: #ffffff;
    }
    /* Header */
    .main-title {
        text-align: center;
        font-size: 40px;
        color: #0040ff91;
        font-weight: bold;
        margin-bottom: 10px;
    }
    .subtitle {
        text-align: center;
        color: #956a04;
        font-size: 18px;
        margin-bottom: 30px;
    }
    /* Tabs */
    div[data-testid="stTabs"] button {
        background-color: #ff000094;
        color: white;
        border-radius: 10px;
      
    }
    div[data-testid="stTabs"] button[aria-selected="true"] {
        background-color: #0040ff91 !important;
        color: black !important;
        font-weight: bold;
        PADDING: 20px;
    }
    /* Recommandations cards */
    .film-card {
        background-color: #00BCD4;
        padding: 15px;
        border-radius: 10px;
        margin-bottom: 10px;
        box-shadow: 0px 0px 10px rgba(255,255,255,0.1);
    }
    </style>
""", unsafe_allow_html=True)


#  1. Chargement des données

@st.cache_data
def charger_donnees():
    path = "D:/mes_travails/powerBI/data"

    ratings = pd.read_csv(f'{path}/ratings.dat', sep='::', engine='python',
                          names=['UserID', 'MovieID', 'Rating', 'Timestamp'])
    movies = pd.read_csv(f'{path}/movies.dat', sep='::', engine='python',
                         names=['MovieID', 'Title', 'Genres'], encoding='latin1')
    users = pd.read_csv(f'{path}/users.dat', sep='::', engine='python',
                        names=['UserID', 'Gender', 'Age', 'Occupation', 'Zip-code'], encoding='latin1')

    movies['Title'] = (movies['Title']
                       .str.lower()
                       .str.replace(r'\(\d{4}\)', '', regex=True)
                       .str.replace(r'[^\w\s]', '', regex=True)
                       .str.strip())

    data = pd.merge(pd.merge(ratings, users), movies)
    data = data.drop_duplicates().reset_index(drop=True)
    genres_dummies = data['Genres'].str.get_dummies(sep='|')
    data = pd.concat([data, genres_dummies], axis=1)
    return data, genres_dummies


@st.cache_data
def preparer_matrices(data, genres_dummies):
    user_movie_matrix = data.pivot_table(index='UserID', columns='Title', values='Rating', fill_value=0)
    user_mean = user_movie_matrix.replace(0, np.nan).mean(axis=1)
    user_matrix_centered = user_movie_matrix.sub(user_mean, axis=0).fillna(0)
    user_similarity = cosine_similarity(user_matrix_centered)
    user_similarity_df = pd.DataFrame(user_similarity, index=user_movie_matrix.index, columns=user_movie_matrix.index)
    movie_features = data[['Title'] + list(genres_dummies.columns)].drop_duplicates('Title').set_index('Title')
    movie_similarity = cosine_similarity(movie_features.values)
    movie_similarity_df = pd.DataFrame(movie_similarity, index=movie_features.index, columns=movie_features.index)
    return user_movie_matrix, user_similarity_df, movie_similarity_df


#  2. Fonctions de recommandation

def recommander_collaboratif(user_id, user_similarity_df, user_movie_matrix, n_similaires=5, n_recommandations=15):
    similar_users = user_similarity_df[user_id].sort_values(ascending=False)[1:n_similaires + 1]
    candidate_movies = user_movie_matrix.loc[similar_users.index]
    weighted_ratings = candidate_movies.T.dot(similar_users) / similar_users.sum()
    rated_movies = user_movie_matrix.loc[user_id]
    weighted_ratings = weighted_ratings[rated_movies == 0]
    return list(weighted_ratings.sort_values(ascending=False).head(n_recommandations).index)


def recommander_contenu(user_id, user_movie_matrix, movie_similarity_df, n_recommandations=15):
    user_rated = user_movie_matrix.loc[user_id]
    liked_movies = user_rated[user_rated >= 4].index
    if len(liked_movies) == 0:
        return []
    scores = movie_similarity_df[liked_movies].sum(axis=1)
    scores = scores[user_rated == 0]
    return list(scores.sort_values(ascending=False).head(n_recommandations).index)


def recommander_films_hybride(user_id, user_movie_matrix, user_similarity_df, movie_similarity_df,
                              alpha=0.6, n_recommandations=15):
    collab = recommander_collaboratif(user_id, user_similarity_df, user_movie_matrix, n_recommandations=n_recommandations)
    contenu = recommander_contenu(user_id, user_movie_matrix, movie_similarity_df, n_recommandations=n_recommandations)
    combined = {}
    for i, movie in enumerate(collab):
        combined[movie] = combined.get(movie, 0) + alpha * (n_recommandations - i) / n_recommandations
    for i, movie in enumerate(contenu):
        combined[movie] = combined.get(movie, 0) + (1 - alpha) * (n_recommandations - i) / n_recommandations
    top_movies = sorted(combined.items(), key=lambda x: x[1], reverse=True)[:n_recommandations]
    return [movie for movie, _ in top_movies]


def recommander_similaires(film_nom, movie_similarity_df, data, n=15):
    film_nom = film_nom.lower().strip()
    if film_nom not in movie_similarity_df.index:
        suggestions = [m for m in movie_similarity_df.index if film_nom in m]
        if len(suggestions) == 0:
            return None, []
        film_nom = suggestions[0]
    similaires = movie_similarity_df[film_nom].sort_values(ascending=False)[1:n+1].index
    resultats = []
    for titre in similaires:
        genres = data.loc[data['Title'] == titre, 'Genres'].iloc[0]
        resultats.append((titre, genres))
    return film_nom, resultats


#  Interface principale

st.markdown("<div class='main-title'>🎬 Recommandation de Films Hybride</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Basé sur la similarité entre utilisateurs et le contenu des films</div>", unsafe_allow_html=True)

data, genres_dummies = charger_donnees()
user_movie_matrix, user_similarity_df, movie_similarity_df = preparer_matrices(data, genres_dummies)

onglet1, onglet2 = st.tabs(["👤 Recommandation personnalisée", "🎞️ Films similaires"])

# --- Onglet 1
with onglet1:
    st.subheader("👤 Recommandation personnalisée")
    user_id = st.selectbox("Sélectionner un utilisateur :", user_movie_matrix.index)
    alpha = st.slider(" Poids du collaboratif (α)", 0.0, 1.0, 0.6, 0.1)

    if st.button("🎥 Obtenir les recommandations", key="reco_user"):
        recommandations = recommander_films_hybride(user_id, user_movie_matrix, user_similarity_df, movie_similarity_df, alpha=alpha)
        if len(recommandations) == 0:
            st.warning("Aucune recommandation disponible.")
        else:
            for film in recommandations:
                genres = data.loc[data['Title'] == film, 'Genres'].iloc[0]
                st.markdown(f"<div class='film-card'><b>{film.title()}</b><br><small>{genres}</small></div>", unsafe_allow_html=True)

# --- Onglet 2
with onglet2:
    st.subheader("🎞️ Trouver des films similaires")
    film_nom = st.text_input("Entrez le nom du film :", "")
    if st.button(" Rechercher", key="reco_film"):
        if film_nom.strip() == "":
            st.warning("Veuillez saisir un nom de film.")
        else:
            film_base, similaires = recommander_similaires(film_nom, movie_similarity_df, data)
            if similaires == []:
                st.error("Film introuvable.")
            else:
                st.success(f"Films similaires à **{film_base.title()}** :")
                for titre, genres in similaires:
                    st.markdown(f"<div class='film-card'><b>{titre.title()}</b><br><small>{genres}</small></div>", unsafe_allow_html=True)


# Sauvegarde des données préparées pour Power BI
data.to_csv("D:/mes_travails/powerBI/export/data_complet.csv", index=False)
user_movie_matrix.to_csv("D:/mes_travails/powerBI/export/user_movie_matrix.csv")
user_similarity_df.to_csv("D:/mes_travails/powerBI/export/user_similarity.csv")

movie_similarity_df.to_csv("D:/mes_travails/powerBI/export/movie_similarity.csv")
