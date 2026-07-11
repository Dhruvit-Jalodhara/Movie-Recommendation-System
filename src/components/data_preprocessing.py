import os
import sys
import pandas as pd
import numpy as np
from dataclasses import dataclass
from scipy.sparse import csr_matrix

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

@dataclass
class DataPreprocessingConfig:
    """
    Configuration data class defining the output paths for the preprocessed artifacts.
    These files will be loaded by the Model Trainer component.
    """
    processed_movies_path: str = os.path.join('artifacts', 'movies_processed.joblib')
    processed_pivot_matrix_path: str = os.path.join('artifacts', 'ratings_pivot.joblib')

class DataPreprocessing:
    """
    Component responsible for cleaning, filtering, and structuring the raw data.
    It applies threshold filters to reduce matrix sparsity, engineers the NLP metadata soup,
    and structures the collaborative user-item pivot matrix.
    """
    def __init__(self):
        """Initializes the DataPreprocessing component with output path configurations."""
        self.preprocessing_config = DataPreprocessingConfig()
        
    def initiate_data_preprocessing(self, movies_path: str, ratings_path: str, tags_path: str):
        """
        Executes the data cleaning and transformation logic strictly following the EDA notebook.
        
        Args:
            movies_path (str): File path to raw movies.csv
            ratings_path (str): File path to raw ratings.csv
            tags_path (str): File path to raw tags.csv
            
        Returns:
            tuple: File paths to the saved (processed_movies, processed_pivot_matrix) artifacts.
            
        Raises:
            CustomException: If an error occurs during the Pandas transformation pipeline.
        """
        logging.info("Starting Data Preprocessing Component...")
        try:
            # 1. Load Data
            logging.info("Loading raw datasets into memory...")
            movies = pd.read_csv(movies_path)
            ratings = pd.read_csv(ratings_path)
            tags = pd.read_csv(tags_path)
            
            # 2. DATA CLEANING & MEMORY OPTIMIZATION
            logging.info("Applying notebook cleaning logic (dropping underscores & downcasting)...")
            movies['movieId'] = movies['movieId'].astype(np.int32)
            
            ratings['userId'] = ratings['userId'].astype(np.int32)
            ratings['movieId'] = ratings['movieId'].astype(np.int32)
            ratings['rating'] = ratings['rating'].astype(np.float32)
            ratings.drop(columns=['timestamp'], inplace=True, errors='ignore')
            
            tags = tags.dropna(subset=['tag'])
            tags['movieId'] = tags['movieId'].astype(np.int32)
            tags['tag'] = tags['tag'].astype(str)
            tags.drop(columns=['timestamp', 'userId'], inplace=True, errors='ignore')

            # 3. ENGINEER NLP METADATA (TAGS + GENRES)
            logging.info("Grouping tags and processing multi-word phrase spacing...")
            
            # Group tags by movieId and join with a pipe character
            grouped_tags = tags.groupby('movieId')['tag'].apply(lambda x: '|'.join(x)).reset_index()
            grouped_tags.rename(columns={'tag': 'tags'}, inplace=True)
            
            # Merge grouped tags into the main movies dataframe
            movies_cleaned = pd.merge(movies, grouped_tags, on='movieId', how='left')
            movies_cleaned['tags'] = movies_cleaned['tags'].fillna('')
            
            # Step 1: Replace internal spaces with underscores to preserve multi-word tags
            movies_cleaned['tags'] = movies_cleaned['tags'].str.replace(' ', '_')
            
            # Step 2: Replace the vertical pipes with regular spaces in both columns
            movies_cleaned['genres'] = movies_cleaned['genres'].str.replace('|', ' ')
            movies_cleaned['tags'] = movies_cleaned['tags'].str.replace('|', ' ')
            
            # Step 3: Smash them together into a single "metadata soup" column
            movies_cleaned['metadata_soup'] = movies_cleaned['genres'] + ' ' + movies_cleaned['tags']
            movies_cleaned['metadata_soup'] = movies_cleaned['metadata_soup'].str.lower().str.strip()

            # 4. APPLY THRESHOLD FILTERING (REDUCE SPARSITY)
            logging.info("Applying >= 50 rating threshold filters for users and movies...")
            
            movie_counts = ratings.groupby('movieId')['rating'].count()
            valid_movies = movie_counts[movie_counts >= 50].index
            df_filtered_movies = ratings[ratings['movieId'].isin(valid_movies)]
            
            user_counts = df_filtered_movies.groupby('userId')['rating'].count()
            valid_users = user_counts[user_counts >= 50].index
            final_ratings = df_filtered_movies[df_filtered_movies['userId'].isin(valid_users)]
            
            logging.info(f"Matrix Sparsity Reduction complete. Final ratings count: {len(final_ratings)}")

            # 5. STRIP ZEROS & COMPRESS TO SPARSE CSR
            logging.info("Building temporary pivot table...")
            pivot_matrix = final_ratings.pivot(index='movieId', columns='userId', values='rating').fillna(0)
            pivot_matrix = pivot_matrix.astype(np.float32)
            
            logging.info("Compressing dense matrix into ultra-light CSR format...")
            sparse_matrix_payload = {
                'csr_matrix': csr_matrix(pivot_matrix.values),
                'movie_ids': pivot_matrix.index.tolist()  # Keeps tracking of matrix row order
            }
            
            # 6. SAVE ARTIFACTS
            logging.info("Saving processed joblib artifacts...")
            save_object(self.preprocessing_config.processed_movies_path, movies_cleaned)
            save_object(self.preprocessing_config.processed_pivot_matrix_path, sparse_matrix_payload)
            
            logging.info("Data Preprocessing completed successfully!")
            
            return (
                self.preprocessing_config.processed_movies_path,
                self.preprocessing_config.processed_pivot_matrix_path
            )
            
        except Exception as e:
            logging.error("Exception encountered during Data Preprocessing.")
            raise CustomException(e, sys)