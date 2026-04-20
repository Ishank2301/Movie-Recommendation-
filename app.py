import os
import pickle
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import requests
import streamlit as st

#  CONFIG
st.set_page_config(page_title="Movie Recommender", page_icon="🎬")

POSTER_BASE_URL = "https://image.tmdb.org/t/p/w500"
DATA_DIR = Path(__file__).resolve().parent

MOVIES_PATH = DATA_DIR / "movies_list.pkl"
SIMILARITY_PATH = DATA_DIR / "similarity.pkl"


# API KEY
def get_tmdb_api_key():
    try:
        return st.secrets["TMDB_API_KEY"]
    except:
        return os.getenv("TMDB_API_KEY", "")


TMDB_API_KEY = get_tmdb_api_key()


# LOAD DATA
@st.cache_resource
def load_data():
    movies = pickle.load(open(MOVIES_PATH, "rb"))
    similarity = pickle.load(open(SIMILARITY_PATH, "rb"))
    return movies, similarity


movies, similarity = load_data()


#  TITLE INDEX MAP
@st.cache_resource
def build_index():
    return {title: i for i, title in enumerate(movies["title"].values)}


title_index = build_index()


# SESSION
@st.cache_resource
def get_session():
    s = requests.Session()
    s.headers.update({"Accept": "application/json"})
    return s


session = get_session()


# FETCH POSTER
@st.cache_data
def fetch_poster(movie_id):
    if not TMDB_API_KEY:
        return None

    try:
        response = session.get(
            f"https://api.themoviedb.org/3/movie/{movie_id}",
            params={"api_key": TMDB_API_KEY, "language": "en-US"},
            timeout=10,
        )
        response.raise_for_status()
        data = response.json()
    except Exception:
        return None

    poster_path = data.get("poster_path")
    if not poster_path:
        return None

    return f"{POSTER_BASE_URL}{poster_path}"


#  MULTI FETCH
def fetch_posters(movie_ids):
    with ThreadPoolExecutor(max_workers=5) as executor:
        return list(executor.map(fetch_poster, movie_ids))


#  RECOMMEND
def recommend(movie):
    idx = title_index.get(movie)

    if idx is None:
        return [], []

    distances = similarity[idx]

    movie_indices = sorted(
        list(enumerate(distances)), reverse=True, key=lambda x: x[1]
    )[1:6]

    names = []
    ids = []

    for i in movie_indices:
        names.append(movies.iloc[i[0]].title)
        ids.append(movies.iloc[i[0]].movie_id)

    return names, ids


#  UI
st.title("🎬 Movie Recommendation System")
st.caption("Get similar movie recommendations instantly")

selected_movie = st.selectbox("Choose a movie", movies["title"].values)

show_posters = st.toggle("Show Posters", value=True)

if not TMDB_API_KEY:
    st.warning("⚠️ TMDB API key not set → posters disabled")

if st.button("Recommend"):
    with st.spinner("Fetching recommendations..."):
        names, ids = recommend(selected_movie)

        if show_posters:
            posters = fetch_posters(ids)
        else:
            posters = [None] * len(names)

        cols = st.columns(5)

        for i in range(len(names)):
            with cols[i]:
                st.subheader(names[i])

                if show_posters and posters[i]:
                    st.image(posters[i], use_container_width=True)
                elif show_posters:
                    st.caption("Poster not available")
