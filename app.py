import os
import pickle
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import requests
import streamlit as st

st.set_page_config(page_title="Movie Recommendation System", page_icon="🎬")


DATA_DIR = Path(__file__).resolve().parent
MOVIES_PATH = DATA_DIR / "movies_list.pkl"
SIMILARITY_PATH = DATA_DIR / "similarity.pkl"
POSTER_BASE_URL = "https://image.tmdb.org/t/p/w500"


def get_tmdb_api_key():
    try:
        return st.secrets["TMDB_API_KEY"].strip()
    except Exception:
        return os.getenv("TMDB_API_KEY", "").strip()


@st.cache_resource
def load_data():
    with MOVIES_PATH.open("rb") as movies_file:
        movies_df = pickle.load(movies_file)

    with SIMILARITY_PATH.open("rb") as similarity_file:
        similarity_matrix = pickle.load(similarity_file)

    return movies_df, similarity_matrix


@st.cache_resource
def build_title_index_lookup():
    return {title: index for index, title in enumerate(movies["title"].values)}


@st.cache_resource
def get_requests_session():
    session = requests.Session()
    session.headers.update({"Accept": "application/json"})
    return session


movies, similarity = load_data()
title_to_index = build_title_index_lookup()
TMDB_API_KEY = get_tmdb_api_key()


@st.cache_data(show_spinner=False)
def fetch_poster(movie_id):
    if not TMDB_API_KEY:
        return None

    try:
        response = get_requests_session().get(
            f"https://api.themoviedb.org/3/movie/{movie_id}",
            params={"api_key": TMDB_API_KEY, "language": "en-US"},
            timeout=2,
        )
        response.raise_for_status()
        data = response.json()
    except requests.RequestException:
        return None

    poster_path = data.get("poster_path")
    if not poster_path:
        return None

    return f"{POSTER_BASE_URL}{poster_path}"


@st.cache_data(show_spinner=False)
def fetch_posters(movie_ids):
    with ThreadPoolExecutor(max_workers=min(5, len(movie_ids))) as executor:
        return list(executor.map(fetch_poster, movie_ids))


@st.cache_data(show_spinner=False)
def recommend(movie):
    movie_index = title_to_index.get(movie)
    if movie_index is None:
        raise ValueError(f"Movie '{movie}' was not found in the dataset.")

    distances = similarity[movie_index]
    candidate_indices = np.argpartition(distances, -6)[-6:]
    sorted_indices = candidate_indices[np.argsort(distances[candidate_indices])[::-1]]
    top_indices = [index for index in sorted_indices if index != movie_index][:5]

    recommended_rows = movies.iloc[top_indices]
    recommendations = recommended_rows["title"].tolist()
    movie_ids = recommended_rows["movie_id"].tolist()
    return recommendations, movie_ids

st.title("Movie Recommendation System")
st.caption("Pick a movie and get the five most similar titles from the dataset.")
st.write("Select a movie below, then click `Recommend` to generate suggestions.")
show_posters = st.toggle("Show posters", value=True)

selected_movie_name = st.selectbox(
    "Choose a movie",
    movies["title"].values,
)

if not TMDB_API_KEY:
    st.info(
        "TMDB poster fetching is disabled because `TMDB_API_KEY` is not set. "
        "Recommendations will still work."
    )
else:
    st.success("TMDB poster fetching is active.")

if st.button("Recommend", type="primary"):
    with st.spinner("Finding similar movies..."):
        try:
            names, movie_ids = recommend(selected_movie_name)
        except ValueError as exc:
            st.error(str(exc))
        else:
            posters = fetch_posters(movie_ids) if show_posters else [None] * len(names)
            st.success(f"Recommendations ready for '{selected_movie_name}'.")
            columns = st.columns(len(names))
            for column, name, poster in zip(columns, names, posters):
                with column:
                    st.subheader(name)
                    if show_posters and poster:
                        st.image(poster, use_container_width=True)
                    elif show_posters:
                        st.caption("Poster unavailable")

            if show_posters and TMDB_API_KEY and all(not poster for poster in posters):
                st.warning(
                    "TMDB poster requests are failing from this machine, so only movie titles are available right now. "
                    "This is likely a network timeout or blocked connection to `api.themoviedb.org`."
                )
