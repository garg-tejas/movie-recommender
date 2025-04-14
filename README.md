# Movie Recommendation System

A content-based movie recommendation system built using Python and Streamlit. The application provides personalized movie recommendations based on:

- Content similarity (plot, keywords, genres)
- Actor filmography
- Director's work
- Genre preferences
- Personalized recommendations based on user preferences

## Features

- **Movie-Based Recommendations**: Find movies similar to ones you enjoy
- **Genre-Based Filtering**: Discover top movies in specific genres with year filtering
- **Actor and Director Exploration**: Find movies by your favorite actors and directors
- **Personalized Recommendations**: Get custom recommendations based on your preferences
- **Auto-complete Suggestions**: User-friendly search with suggestions
- **Robust Error Handling**: Failsafe display methods ensure recommendations are visible even with dependency issues
- **Custom Dataset Support**: Option to upload your own movie dataset

## How It Works

The recommendation system uses TF-IDF (Term Frequency-Inverse Document Frequency) vectorization and cosine similarity to find movies with similar content features. The system analyzes movie attributes such as:

- Plot summaries
- Genres
- Directors
- Cast members
- Keywords
- Production companies

The recommendation algorithm creates a "content soup" for each movie by combining these features with appropriate weighting. For example, genres and directors are given more weight than other features to ensure relevance.

## Using the App

1. The app loads with a default movie dataset (`cleaned_movies.csv`)
2. Use the sidebar to upload a custom dataset (optional)
3. Navigate through the tabs to get different types of recommendations:
   - **Movie Based**: Find similar movies based on content
   - **Genre Based**: Browse top movies in specific genres with year filtering
   - **Actor/Director**: Discover films by favorite actors or directors
   - **Personalized**: Get recommendations based on your preferences
   - **Search**: Find movies by title or keywords
4. Search for movies, actors, or directors using the autocomplete search boxes
5. Build a personalized recommendation profile by selecting movies you like/dislike

## Dataset Format

The app works with CSV files containing movie data with the following columns:

- **Required**: id, title
- **Recommended**: genres, directors, cast, overview, vote_average, vote_count, release_date, year, keywords
- **Optional**: popularity, production_companies, tagline

The default dataset (`cleaned_movies.csv`) includes a comprehensive collection of movies with all recommended fields.

## File Structure

```
movie-recommendation/
├── app.py                   # Main Streamlit application
├── recommender.py           # Recommendation engine logic
├── requirements.txt         # Python dependencies
├── packages.txt             # System dependencies
├── README.md                # Documentation
└── data/                    # Data files
    ├── cleaned_movies.csv   # Default movie dataset
    ├── unique_cast.txt      # Autocomplete data for actors
    ├── unique_directors.txt # Autocomplete data for directors
    ├── unique_genres.txt    # Autocomplete data for genres
    ├── unique_keywords.txt  # Autocomplete data for keywords
    └── ...                  # Other autocomplete data files
```

## Technology Stack

- **Python**: Core programming language
- **Pandas**: Data manipulation and data analysis
- **Scikit-learn**: TF-IDF vectorization and cosine similarity calculations
- **Streamlit**: Web application framework for interactive interface
- **PyArrow** (optional): For optimized dataframe handling

## Deployment

This application is designed to be easily deployed on Streamlit Cloud:

1. Connect your GitHub repository to Streamlit Cloud
2. Select the repository containing this code
3. Set the main file path to `app.py`
4. Deploy!

The app automatically handles dependencies and data loading. The default dataset (`cleaned_movies.csv`) is included in the repository for immediate use.

## Error Handling

The application includes robust error handling to ensure reliability:

- Fallback display methods if PyArrow is not available
- Graceful handling of missing data fields
- Informative user messages for search results
- Automatic recovery for file operations
