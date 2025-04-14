import streamlit as st
import pandas as pd
import time
import numpy as np
import os
from recommender import MovieRecommender

try:
    import pyarrow
except ImportError:
    print("Warning: PyArrow import failed. Using pandas display fallback.")

    st.set_option('deprecation.showPyplotGlobalUse', False)

def safe_display_table(df):
    """Display a dataframe as a table, with fallback if PyArrow fails."""
    try:
        st.table(df)
    except Exception as e:
        print(f"Standard table display failed: {e}")
        st.write("Recommendations:")
        st.dataframe(df)
        try:
            st.write(df.to_html(escape=False), unsafe_allow_html=True)
        except:
            st.write(df.values)
    
st.set_page_config(
    page_title="Movie Recommendation System",
    page_icon="🎬",
    layout="wide"
)

def load_autocomplete_data(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return [line.strip() for line in f if line.strip()]
    except FileNotFoundError:
        return []

if 'recommender' not in st.session_state:
    st.session_state.recommender = None
if 'dataset_loaded' not in st.session_state:
    st.session_state.dataset_loaded = False
if 'movies_df' not in st.session_state:
    st.session_state.movies_df = None
if 'suggestions' not in st.session_state:
    st.session_state.suggestions = {
        'titles': [],
        'directors': [],
        'actors': [],
        'genres': [],
        'keywords': [],
        'writers': [],
        'production_companies': []
    }

st.title("🎬 Movie Recommendation System")
st.markdown("""
This app provides movie recommendations based on content similarity, 
genre, actors, directors, and personalized preferences.
""")

def load_and_prepare_data(file_path):
    with st.spinner("Loading dataset and building recommendation model..."):
        recommender = MovieRecommender()
        
        st.session_state.movies_df = recommender.load_data(file_path)
        
        recommender.clean_data()
        
        recommender.create_content_soup()
        
        recommender.create_tfidf_matrix()
        
        st.session_state.suggestions['titles'] = load_autocomplete_data("data/unique_title.txt")
        st.session_state.suggestions['directors'] = load_autocomplete_data("data/unique_directors.txt")
        st.session_state.suggestions['actors'] = load_autocomplete_data("data/unique_cast.txt")
        st.session_state.suggestions['genres'] = load_autocomplete_data("data/unique_genres.txt")
        st.session_state.suggestions['keywords'] = load_autocomplete_data("data/unique_keywords.txt")
        st.session_state.suggestions['writers'] = load_autocomplete_data("data/unique_writers.txt")
        st.session_state.suggestions['production_companies'] = load_autocomplete_data("data/unique_production_companies.txt")
        
        st.session_state.recommender = recommender
        st.session_state.dataset_loaded = True
        
    st.success("Dataset loaded and model built successfully!")

with st.sidebar:
    st.header("Dataset")
    default_dataset = "data/cleaned_movies.csv"
    custom_dataset = st.checkbox("Upload custom dataset", value=False)
    
    if custom_dataset:
        dataset_file = st.file_uploader("Upload your movie dataset CSV", type=['csv'])
        
        if dataset_file is not None:
            with open("temp_dataset.csv", "wb") as f:
                f.write(dataset_file.getbuffer())
            
            if not st.session_state.dataset_loaded or st.session_state.current_dataset != "temp_dataset.csv":
                st.session_state.current_dataset = "temp_dataset.csv"
                load_and_prepare_data("temp_dataset.csv")
    
    elif not st.session_state.dataset_loaded or getattr(st.session_state, 'current_dataset', '') != default_dataset:
        st.session_state.current_dataset = default_dataset
        load_and_prepare_data(default_dataset)
    
    if st.session_state.movies_df is not None:
        st.subheader("Dataset Information")
        st.write(f"Total Movies: {len(st.session_state.movies_df)}")
        
        available_columns = st.session_state.movies_df.columns.tolist()
        st.write("Available Features:", ", ".join(available_columns[:5]) + "...")
        
        if 'genres' in st.session_state.movies_df.columns:
            genres = set()
            for g in st.session_state.movies_df['genres'].str.split(',').dropna():
                genres.update([genre.strip() for genre in g])
            st.write(f"Number of Genres: {len(genres)}")
        
        if 'year' in st.session_state.movies_df.columns:
            years = st.session_state.movies_df['year'].dropna().astype(int)
            st.write(f"Year Range: {int(years.min())} - {int(years.max())}")

if st.session_state.dataset_loaded:
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Movie Based", "Genre Based", "Actor/Director", "Personalized", "Search"
    ])
    
    with tab1:
        st.header("Find Similar Movies")
        st.write("Get movie recommendations based on movies you enjoyed")
        
        selected_movie = st.selectbox(
            "Select or type a movie title:", 
            options=st.session_state.suggestions['titles'],
            key="movie_selectbox"
        )
        
        if st.button("Get Similar Movies", key="similar_movies_btn"):
            if selected_movie:
                with st.spinner("Finding similar movies..."):
                    recommendations = st.session_state.recommender.get_recommendations(selected_movie)
                    
                if not recommendations.empty:
                    st.subheader(f"Movies similar to '{selected_movie}':")
                    safe_display_table(recommendations)
                else:
                    st.warning("No similar movies found. Try another title.")
    
    with tab2:
        st.header("Browse by Genre")
        st.write("Discover top rated movies in specific genres")
        
        col1, col2 = st.columns(2)
        
        with col1:
            selected_genre = st.selectbox(
                "Select a genre:", 
                options=sorted(st.session_state.suggestions['genres']),
                key="genre_selectbox"
            )
        
        with col2:
            year_range = st.slider(
                "Filter by year:", 
                min_value=int(st.session_state.movies_df['year'].min()) if 'year' in st.session_state.movies_df.columns else 1900,
                max_value=int(st.session_state.movies_df['year'].max()) if 'year' in st.session_state.movies_df.columns else 2023,
                value=(
                    int(st.session_state.movies_df['year'].min()) if 'year' in st.session_state.movies_df.columns else 1900,
                    int(st.session_state.movies_df['year'].max()) if 'year' in st.session_state.movies_df.columns else 2023
                )
            )
        
        if st.button("Find Movies", key="genre_movies_btn"):
            if selected_genre:
                with st.spinner("Finding movies..."):
                    recommendations = st.session_state.recommender.get_recommendations_by_genre(
                        selected_genre, 
                        min_year=year_range[0], 
                        max_year=year_range[1]
                    )
                    
                if not recommendations.empty:
                    st.subheader(f"Top {selected_genre} movies ({year_range[0]}-{year_range[1]}):")
                    safe_display_table(recommendations)
                else:
                    st.warning(f"No movies found in the {selected_genre} genre for the selected year range.")
    
    with tab3:
        st.header("Find by Actor or Director")
        st.write("Discover movies by your favorite actors or directors")
        
        search_type = st.radio("Search by:", ("Actor", "Director"))
        
        if search_type == "Actor":
            selected_person = st.selectbox(
                "Select or type an actor name:", 
                options=st.session_state.suggestions['actors'],
                key="actor_selectbox"
            )
            
            if st.button("Find Actor's Movies", key="actor_movies_btn"):
                if selected_person:
                    with st.spinner("Finding movies..."):
                        recommendations = st.session_state.recommender.get_recommendations_by_actor(selected_person)
                        
                    if not recommendations.empty:
                        st.subheader(f"Movies starring {selected_person}:")
                        safe_display_table(recommendations)
                    else:
                        st.warning(f"No movies found with {selected_person}.")
        else:
            selected_person = st.selectbox(
                "Select or type a director name:", 
                options=st.session_state.suggestions['directors'],
                key="director_selectbox"
            )
            
            if st.button("Find Director's Movies", key="director_movies_btn"):
                if selected_person:
                    with st.spinner("Finding movies..."):
                        recommendations = st.session_state.recommender.get_recommendations_by_director(selected_person)
                        
                    if not recommendations.empty:
                        st.subheader(f"Movies directed by {selected_person}:")
                        safe_display_table(recommendations)
                    else:
                        st.warning(f"No movies found directed by {selected_person}.")
    
    with tab4:
        st.header("Personalized Recommendations")
        st.write("Get personalized movie recommendations based on your preferences")
        
        liked_movies = st.multiselect(
            "Select movies you like:", 
            options=st.session_state.suggestions['titles'],
            key="liked_movies"
        )
        
        disliked_movies = st.multiselect(
            "Select movies you dislike (optional):", 
            options=st.session_state.suggestions['titles'],
            key="disliked_movies"
        )
        
        preferred_genres = st.multiselect(
            "Select genres you enjoy (optional):", 
            options=sorted(st.session_state.suggestions['genres']),
            key="preferred_genres"
        )
        
        preferred_directors = st.multiselect(
            "Select directors you like (optional):", 
            options=st.session_state.suggestions['directors'],
            key="preferred_directors"
        )
        
        if st.button("Get Personalized Recommendations", key="personalized_btn"):
            if liked_movies:
                with st.spinner("Generating personalized recommendations..."):
                    recommendations = st.session_state.recommender.get_personalized_recommendations(
                        liked_movies=liked_movies,
                        disliked_movies=disliked_movies if disliked_movies else None,
                        genres_preference=preferred_genres if preferred_genres else None,
                        directors_preference=preferred_directors if preferred_directors else None
                    )
                    
                if not recommendations.empty:
                    st.subheader("Your Personalized Recommendations:")
                    safe_display_table(recommendations)
                else:
                    st.warning("Couldn't generate personalized recommendations with current preferences.")
            else:
                st.warning("Please select at least one movie you like.")
    
    with tab5:
        st.header("Search Movies")
        st.write("Search for movies by title or keywords")
        
        search_option = st.radio("Search by:", ("Title", "Keywords"))
        
        if search_option == "Title":
            title_query = st.text_input("Enter movie title to search:", key="title_search")
            
            if st.button("Search", key="title_search_btn"):
                if title_query:
                    with st.spinner("Searching..."):
                        results = st.session_state.recommender.find_movies_by_title(title_query)
                        
                    if not results.empty:
                        st.subheader(f"Movies matching '{title_query}':")
                        safe_display_table(results)
                    else:
                        st.warning(f"No movies found matching '{title_query}'.")
                else:
                    st.warning("Please enter a title to search.")
        else:
            keywords = st.text_input("Enter keywords (comma separated):", key="keyword_search")
            
            if st.button("Search", key="keyword_search_btn"):
                if keywords:
                    keyword_list = [k.strip() for k in keywords.split(',') if k.strip()]
                    
                    if keyword_list:
                        with st.spinner("Searching..."):
                            results = st.session_state.recommender.get_recommendations_by_keywords(keyword_list)
                            
                        if not results.empty:
                            st.subheader(f"Movies matching keywords '{keywords}':")
                            safe_display_table(results)
                        else:
                            st.warning(f"No movies found with keywords '{keywords}'.")
                    else:
                        st.warning("Please enter valid keywords.")
                else:
                    st.warning("Please enter keywords to search.")
else:
    st.info("Please wait for the dataset to load completely...")