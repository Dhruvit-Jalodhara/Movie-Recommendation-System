import os
import sys
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity, linear_kernel
from dataclasses import dataclass

from src.exception import CustomException
from src.logger import logging
from src.utils import load_object

@dataclass
class PredictPipelineConfig:
    """
    Configuration dataclass defining input paths for the precomputed feature spaces
    and mapping indexes required to execute live hybrid inference.
    """
    processed_movies_path: str = os.path.join('artifacts', 'movies_processed.joblib')
    movie_features_svd_path: str = os.path.join("artifacts", "movie_features_svd.joblib")
    tfidf_matrix_path: str = os.path.join("artifacts", "tfidf_matrix.joblib")
    movie_mapping_path: str = os.path.join("artifacts", "movie_mapping.joblib")

class PredictPipeline:
    """
    Inference pipeline responsible for executing real-time hybrid movie recommendations.
    It combines behavioral collaborative patterns (SVD) and contextual metadata soup (TF-IDF)
    using a dynamically adjustable linear combination balancer.
    """
    def __init__(self):
        """Initializes the pipeline and loads all required vector components into memory."""
        try:
            self.config = PredictPipelineConfig()
            
            logging.info("Loading evaluation feature spaces for real-time inference...")
            self.movies_df = load_object(self.config.processed_movies_path)
            self.movie_features_svd = load_object(self.config.movie_features_svd_path)
            self.tfidf_matrix = load_object(self.config.tfidf_matrix_path)
            self.movie_to_idx = load_object(self.config.movie_mapping_path)
            
            # Fast reverse lookup dictionary mapping index positions back to titles
            self.idx_to_movie = {v: k for k, v in self.movie_to_idx.items()}
            logging.info("Live inference structures ready!")
            
        except Exception as e:
            raise CustomException(e, sys)

    def get_hybrid_recommendations(self, movie_title: str, alpha: float = 0.5, top_n: int = 5):
        """
        Blends collaborative scores and text similarity vectors linearly to yield recommendations.
        Features automatic index space mapping to align collaborative and text models perfectly.
        """
        try:
            # 1. Confirm the movie exists within our collaborative catalog universe
            if movie_title not in self.movie_to_idx.index:
                logging.warning(f"Query target '{movie_title}' not tracked in matrix parameters.")
                return None

            # 2. Extract internal collaborative positional vector row location
            movie_idx = self.movie_to_idx[movie_title]
            
            # Compute Collaborative Scoring (16,116 values)
            target_svd_vector = self.movie_features_svd[movie_idx].reshape(1, -1)
            coll_scores = cosine_similarity(target_svd_vector, self.movie_features_svd).flatten()
            
            # 3. Dynamic Alignment: Find the true TF-IDF row index for the query movie
            target_tfidf_idx = movie_idx # Fallback default
            
            if len(self.movies_df) == self.tfidf_matrix.shape[0] and 'title' in self.movies_df.columns:
                matching_rows = self.movies_df[self.movies_df['title'].str.lower() == movie_title.lower()].index
                if len(matching_rows) > 0:
                    target_tfidf_idx = matching_rows[0]
            elif len(self.movies_df) == len(coll_scores):
                if 'tfidf_idx' in self.movies_df.columns:
                    target_tfidf_idx = int(self.movies_df.loc[movie_idx, 'tfidf_idx'])
                elif 'index' in self.movies_df.columns:
                    target_tfidf_idx = int(self.movies_df.loc[movie_idx, 'index'])

            # Compute raw text similarity scores across all 86,537 items
            target_tfidf_vector = self.tfidf_matrix[target_tfidf_idx]
            raw_content_scores = linear_kernel(target_tfidf_vector, self.tfidf_matrix).flatten()
            
            # 4. Map the 86,537 content scores into an array aligned with the 16,116 catalog items
            aligned_content_scores = np.zeros_like(coll_scores)
            
            if len(self.movies_df) == self.tfidf_matrix.shape[0] and 'title' in self.movies_df.columns:
                # Optimize title mapping via an on-demand dictionary lookup cache
                if not hasattr(self, '_title_to_tfidf_cache'):
                    self._title_to_tfidf_cache = {str(t).lower().strip(): idx for idx, t in enumerate(self.movies_df['title'])}
                
                for i in range(len(coll_scores)):
                    current_title = str(self.idx_to_movie[i]).lower().strip()
                    if current_title in self._title_to_tfidf_cache:
                        aligned_content_scores[i] = raw_content_scores[self._title_to_tfidf_cache[current_title]]
                        
            elif len(self.movies_df) == len(coll_scores):
                for i in range(len(coll_scores)):
                    orig_idx = i
                    if 'tfidf_idx' in self.movies_df.columns:
                        orig_idx = int(self.movies_df.loc[i, 'tfidf_idx'])
                    elif 'index' in self.movies_df.columns:
                        orig_idx = int(self.movies_df.loc[i, 'index'])
                    
                    if orig_idx < len(raw_content_scores):
                        aligned_content_scores[i] = raw_content_scores[orig_idx]
            else:
                # Safe structural cutoff fallback if metadata indexes are missing
                min_len = min(len(coll_scores), len(raw_content_scores))
                aligned_content_scores[:min_len] = raw_content_scores[:min_len]
            
            # 5. Execute Linear Combination Blending using perfectly aligned vectors
            hybrid_scores = (alpha * coll_scores) + ((1 - alpha) * aligned_content_scores)
            
            # 6. Extract high-to-low sorted array coordinates
            sorted_indices = np.argsort(hybrid_scores)[::-1]
            
            recommendations = []
            rank = 1
            
            for idx in sorted_indices:
                # Shield the algorithm from recommending the query movie back to the user
                if idx == movie_idx:
                    continue
                    
                recommended_title = self.idx_to_movie[idx]
                
                recommendations.append({
                    "rank": rank,
                    "title": recommended_title,
                    "combined_score": round(float(hybrid_scores[idx]) * 100, 2),
                    "svd_score": round(float(coll_scores[idx]) * 100, 2),
                    "tfidf_score": round(float(aligned_content_scores[idx]) * 100, 2)
                })
                
                rank += 1
                if rank > top_n:
                    break
                    
            return recommendations

        except Exception as e:
            logging.error("Prediction breakdown caught inside pipeline engine.")
            raise CustomException(e, sys)