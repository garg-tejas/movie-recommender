import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
from datetime import datetime
import kagglehub
from kagglehub import KaggleDatasetAdapter

class MovieRecommender:
    def __init__(self):
        self.movies_df = None
        self.tfidf_matrix = None
        self.cosine_sim = None
        self.indices = None
        
    def load_data(self, file_path):
        """Load movie data from CSV file"""
        self.movies_df = pd.read_csv(file_path)
        print(f"Loaded {len(self.movies_df)} movies")
        
        if 'release_date' in self.movies_df.columns:
            self.movies_df['release_date'] = pd.to_datetime(self.movies_df['release_date'], errors='coerce')
            self.movies_df['year'] = self.movies_df['release_date'].dt.year
        
        return self.movies_df
    
    def clean_data(self):
        """Clean and prepare the dataset"""
        df = self.movies_df.copy()
        
        df = df.dropna()
        
        text_columns = ['title', 'overview', 'tagline', 'genres', 'keywords', 
                         'directors', 'writers', 'cast', 'production_companies']
        
        for col in text_columns:
            if col in df.columns:
                df[col] = df[col].astype(str)
        
        self.movies_df = df
        print(f"Cleaned data: {len(self.movies_df)} movies remain after dropping missing values")
        return df
    
    def create_content_soup(self):
        """Create a text representation combining multiple features"""
        df = self.movies_df.copy()
        
        df['content_soup'] = ''
        
        if 'genres' in df.columns:
            df['content_soup'] += df['genres'].apply(lambda x: ' '.join([x.lower()] * 3))
        
        if 'directors' in df.columns:
            df['content_soup'] += ' ' + df['directors'].apply(lambda x: ' '.join([x.lower()] * 3))
        
        if 'cast' in df.columns:
            df['content_soup'] += ' ' + df['cast'].apply(
                lambda x: ' '.join(x.lower().split(',')[:3] * 2) + ' ' + ' '.join(x.lower().split(',')[3:]))
        
        if 'keywords' in df.columns:
            df['content_soup'] += ' ' + df['keywords'].apply(lambda x: x.lower())
        
        if 'overview' in df.columns:
            df['content_soup'] += ' ' + df['overview'].apply(lambda x: x.lower())
        
        if 'writers' in df.columns:
            df['content_soup'] += ' ' + df['writers'].apply(lambda x: x.lower())
        
        if 'original_language' in df.columns:
            df['content_soup'] += ' ' + df['original_language'].apply(lambda x: x.lower())
        
        df['content_soup'] = df['content_soup'].apply(lambda x: re.sub(r'[^\w\s]', ' ', x))
        
        self.movies_df = df
        
        self.indices = pd.Series(df.index, index=df['title']).drop_duplicates()
        
        return df
    
    def create_tfidf_matrix(self):
        """Create TF-IDF matrix from content soup"""
        tfidf = TfidfVectorizer(stop_words='english', max_features=5000)
        
        self.tfidf_matrix = tfidf.fit_transform(self.movies_df['content_soup'])

        self.cosine_sim = cosine_similarity(self.tfidf_matrix, self.tfidf_matrix)
        
        print(f"Created content matrix with shape: {self.tfidf_matrix.shape}")
        return self.cosine_sim
    
    def get_recommendations(self, title, n_recommendations=10):
        """Get movie recommendations based on content similarity"""
        if title not in self.indices:
            close_matches = self.movies_df[self.movies_df['title'].str.contains(title, case=False)]
            if len(close_matches) > 0:
                print(f"Movie '{title}' not found. Did you mean one of these?")
                print(close_matches['title'].head().tolist())
            else:
                print(f"Movie '{title}' not found in the dataset.")
            return pd.DataFrame()
        
        idx = self.indices[title]
        
        sim_scores = list(enumerate(self.cosine_sim[idx]))
        
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        
        sim_scores = sim_scores[1:n_recommendations+1]
        
        movie_indices = [i[0] for i in sim_scores]
        
        columns_to_include = ['id', 'title', 'genres', 'directors', 'vote_average', 
                             'vote_count', 'release_date', 'popularity']
        available_columns = [col for col in columns_to_include if col in self.movies_df.columns]
        
        recommendations = self.movies_df.iloc[movie_indices][available_columns].copy()
        
        recommendations['similarity_score'] = [i[1] for i in sim_scores]
        
        recommendations = recommendations.sort_values('similarity_score', ascending=False)
        
        return recommendations
    
    def get_recommendations_by_actor(self, actor_name, n_recommendations=10):
        """Get movie recommendations with a specific actor"""
        if 'cast' not in self.movies_df.columns:
            print("Cast information not available in dataset")
            return pd.DataFrame()
        
        actor_movies = self.movies_df[self.movies_df['cast'].str.contains(actor_name, case=False, na=False)]
        
        if len(actor_movies) == 0:
            print(f"No movies found with actor '{actor_name}'")
            return pd.DataFrame()
        
        if 'vote_average' in self.movies_df.columns and 'vote_count' in self.movies_df.columns:
            min_votes = 50
            qualified_movies = actor_movies[actor_movies['vote_count'] >= min_votes]
            
            if len(qualified_movies) > 0:
                qualified_movies['score'] = qualified_movies['vote_average'] * qualified_movies['vote_count'] / qualified_movies['vote_count'].max()
                recommendations = qualified_movies.sort_values('score', ascending=False).head(n_recommendations)
            else:
                recommendations = actor_movies.sort_values('vote_average', ascending=False).head(n_recommendations)
        elif 'popularity' in self.movies_df.columns:
            recommendations = actor_movies.sort_values('popularity', ascending=False).head(n_recommendations)
        else:
            recommendations = actor_movies.head(n_recommendations)
        
        columns_to_include = ['id', 'title', 'genres', 'directors', 'vote_average', 
                             'vote_count', 'release_date', 'popularity']
        available_columns = [col for col in columns_to_include if col in self.movies_df.columns]
        
        return recommendations[available_columns]
    
    def get_recommendations_by_director(self, director_name, n_recommendations=10):
        """Get movie recommendations by a specific director"""
        if 'directors' not in self.movies_df.columns:
            print("Director information not available in dataset")
            return pd.DataFrame()
        
        director_movies = self.movies_df[self.movies_df['directors'].str.contains(director_name, case=False, na=False)]
        
        if len(director_movies) == 0:
            print(f"No movies found with director '{director_name}'")
            return pd.DataFrame()
        
        if 'vote_average' in self.movies_df.columns and 'vote_count' in self.movies_df.columns:
            min_votes = 20
            qualified_movies = director_movies[director_movies['vote_count'] >= min_votes]
            
            if len(qualified_movies) > 0:
                recommendations = qualified_movies.sort_values('vote_average', ascending=False).head(n_recommendations)
            else:
                recommendations = director_movies.sort_values('vote_average', ascending=False).head(n_recommendations)
        else:
            recommendations = director_movies.head(n_recommendations)
        
        columns_to_include = ['id', 'title', 'genres', 'vote_average', 
                             'vote_count', 'release_date', 'popularity']
        available_columns = [col for col in columns_to_include if col in self.movies_df.columns]
        
        return recommendations[available_columns]
    
    def get_recommendations_by_genre(self, genre, min_year=None, max_year=None, n_recommendations=10):
        """Get top rated movies in a specific genre with optional year filtering"""
        if 'genres' not in self.movies_df.columns:
            print("Genre information not available in dataset")
            return pd.DataFrame()
        
        genre_movies = self.movies_df[self.movies_df['genres'].str.contains(genre, case=False, na=False)]
        
        if min_year is not None and 'year' in self.movies_df.columns:
            genre_movies = genre_movies[genre_movies['year'] >= min_year]
        
        if max_year is not None and 'year' in self.movies_df.columns:
            genre_movies = genre_movies[genre_movies['year'] <= max_year]
        
        if len(genre_movies) == 0:
            print(f"No movies found with genre '{genre}'")
            return pd.DataFrame()
        
        if 'vote_average' in self.movies_df.columns and 'vote_count' in self.movies_df.columns:
            min_votes = 100
            qualified_movies = genre_movies[genre_movies['vote_count'] >= min_votes].copy()
            
            if len(qualified_movies) > 0:
                C = qualified_movies['vote_average'].mean()
                m = min_votes
                
                qualified_movies.loc[:, 'weighted_rating'] = (
                    (qualified_movies['vote_count'] / (qualified_movies['vote_count'] + m)) * qualified_movies['vote_average'] + 
                    (m / (qualified_movies['vote_count'] + m)) * C
                )
                
                recommendations = qualified_movies.sort_values('weighted_rating', ascending=False).head(n_recommendations)
            else:
                recommendations = genre_movies.sort_values('popularity', ascending=False).head(n_recommendations)
        elif 'popularity' in self.movies_df.columns:
            recommendations = genre_movies.sort_values('popularity', ascending=False).head(n_recommendations)
        else:
            recommendations = genre_movies.head(n_recommendations)
        
        columns_to_include = ['id', 'title', 'genres', 'directors', 'vote_average', 
                             'vote_count', 'release_date', 'popularity']
        available_columns = [col for col in columns_to_include if col in self.movies_df.columns]
        
        return recommendations[available_columns]
    
    def get_recommendations_by_keywords(self, keywords, n_recommendations=10):
        """Get movies matching specific keywords"""
        if 'keywords' not in self.movies_df.columns and 'overview' not in self.movies_df.columns:
            print("Keyword information not available in dataset")
            return pd.DataFrame()
        
        df = self.movies_df.copy()
        search_field = ''
        
        if 'keywords' in df.columns:
            search_field += df['keywords'].fillna('')
        
        if 'overview' in df.columns:
            search_field += ' ' + df['overview'].fillna('')
        
        keyword_pattern = '|'.join(keywords)
        matching_movies = df[search_field.str.contains(keyword_pattern, case=False, na=False)]
        
        if len(matching_movies) == 0:
            print(f"No movies found with keywords: {keywords}")
            return pd.DataFrame()
        
        if 'vote_average' in df.columns and 'vote_count' in df.columns:
            min_votes = 50
            qualified_movies = matching_movies[matching_movies['vote_count'] >= min_votes]
            
            if len(qualified_movies) > 0:
                recommendations = qualified_movies.sort_values('vote_average', ascending=False).head(n_recommendations)
            else:
                recommendations = matching_movies.sort_values('vote_average', ascending=False).head(n_recommendations)
        elif 'popularity' in df.columns:
            recommendations = matching_movies.sort_values('popularity', ascending=False).head(n_recommendations)
        else:
            recommendations = matching_movies.head(n_recommendations)
        
        columns_to_include = ['id', 'title', 'genres', 'directors', 'vote_average', 
                             'vote_count', 'release_date', 'popularity', 'overview']
        available_columns = [col for col in columns_to_include if col in df.columns]
        
        return recommendations[available_columns]
    
    def get_personalized_recommendations(self, liked_movies, disliked_movies=None, genres_preference=None, 
                                        directors_preference=None, n_recommendations=10):
        """Get personalized recommendations based on user preferences"""
        if not liked_movies:
            print("Please provide at least one liked movie")
            return pd.DataFrame()
        
        all_movies = {idx: 0 for idx in self.movies_df.index}
        
        for movie in liked_movies:
            if movie in self.indices:
                idx = self.indices[movie]
                sim_scores = list(enumerate(self.cosine_sim[idx]))
                
                for i, score in sim_scores:
                    if i in all_movies:
                        all_movies[i] += score

        if disliked_movies:
            for movie in disliked_movies:
                if movie in self.indices:
                    idx = self.indices[movie]
                    sim_scores = list(enumerate(self.cosine_sim[idx]))
                    
                    for i, score in sim_scores:
                        if i in all_movies:
                            all_movies[i] -= score
        
        if genres_preference and 'genres' in self.movies_df.columns:
            for i in all_movies.keys():
                if i in self.movies_df.index:
                    row = self.movies_df.loc[i]
                    for genre in genres_preference:
                        if genre.lower() in row['genres'].lower():
                            all_movies[i] += 0.5
        
        if directors_preference and 'directors' in self.movies_df.columns:
            for i in all_movies.keys():
                if i in self.movies_df.index:
                    row = self.movies_df.loc[i]
                    for director in directors_preference:
                        if director.lower() in row['directors'].lower():
                            all_movies[i] += 0.7
        
        sorted_scores = sorted(all_movies.items(), key=lambda x: x[1], reverse=True)
        
        liked_indices = []
        try:
            liked_indices = [self.indices[m] for m in liked_movies if m in self.indices]
        except:
            pass
            
        movie_indices = []
        count = 0
        for i, score in sorted_scores:
            if i not in liked_indices and count < n_recommendations:
                movie_indices.append(i)
                count += 1
            if count >= n_recommendations:
                break
        
        columns_to_include = ['id', 'title', 'genres', 'directors', 'vote_average', 
                             'vote_count', 'release_date', 'popularity']
        available_columns = [col for col in columns_to_include if col in self.movies_df.columns]
        
        if not movie_indices:
            print("No recommendations found with current preferences")
            return pd.DataFrame()
            
        recommendations = self.movies_df.loc[movie_indices, available_columns].copy()
        
        scores = [all_movies[idx] for idx in movie_indices]
        recommendations['rec_score'] = scores
        
        recommendations = recommendations.sort_values('rec_score', ascending=False)
        
        return recommendations
    
    def get_similar_movies_hybrid(self, movie_title, n_recommendations=10):
        """Get similar movies using a hybrid of content and metadata"""
        if movie_title not in self.indices:
            print(f"Movie '{movie_title}' not found in the dataset")
            return pd.DataFrame()
        
        movie_idx = self.indices[movie_title]
        movie_details = self.movies_df.iloc[movie_idx]
        
        sim_scores = list(enumerate(self.cosine_sim[movie_idx]))
        
        movie_genres = movie_details['genres'].lower() if 'genres' in self.movies_df.columns else ''
        movie_director = movie_details['directors'].lower() if 'directors' in self.movies_df.columns else ''
        movie_year = movie_details['year'] if 'year' in self.movies_df.columns else None
        
        hybrid_scores = []
        for idx, content_score in sim_scores:
            if idx == movie_idx:
                continue
                
            hybrid_score = content_score
            
            if 'genres' in self.movies_df.columns and movie_genres:
                curr_genres = self.movies_df.iloc[idx]['genres'].lower()
                for genre in movie_genres.split(','):
                    if genre.strip() in curr_genres:
                        hybrid_score += 0.2
            
            if 'directors' in self.movies_df.columns and movie_director:
                curr_director = self.movies_df.iloc[idx]['directors'].lower()
                if movie_director in curr_director:
                    hybrid_score += 0.3
            
            if 'year' in self.movies_df.columns and movie_year:
                curr_year = self.movies_df.iloc[idx]['year']
                if curr_year and abs(curr_year - movie_year) <= 5:
                    hybrid_score += 0.1
            
            hybrid_scores.append((idx, hybrid_score))
        
        hybrid_scores = sorted(hybrid_scores, key=lambda x: x[1], reverse=True)
        
        top_indices = [i[0] for i in hybrid_scores[:n_recommendations]]
        
        columns_to_include = ['id', 'title', 'genres', 'directors', 'vote_average', 
                             'vote_count', 'release_date', 'popularity', 'overview']
        available_columns = [col for col in columns_to_include if col in self.movies_df.columns]
        
        recommendations = self.movies_df.iloc[top_indices][available_columns].copy()

        recommendations['hybrid_score'] = [i[1] for i in hybrid_scores[:n_recommendations]]
        
        return recommendations
    
    def find_movies_by_title(self, title_query):
        """Search for movies by title"""
        matching_movies = self.movies_df[self.movies_df['title'].str.contains(title_query, case=False, na=False)]
        
        if len(matching_movies) == 0:
            print(f"No movies found matching '{title_query}'")
            return pd.DataFrame()
        
        columns_to_include = ['id', 'title', 'genres', 'directors', 'vote_average', 
                             'vote_count', 'release_date', 'popularity']
        available_columns = [col for col in columns_to_include if col in self.movies_df.columns]
        
        return matching_movies[available_columns].head(10)