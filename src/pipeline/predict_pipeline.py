import os
import sys
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity, linear_kernel
from dataclasses import dataclass
import joblib

# Import custom monitoring and diagnostic tools from your framework
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
            processed_movies_path = os.path.join('artifacts', 'movies_processed.joblib')
            movie_features_svd_path = os.path.join("artifacts", "movie_features_svd.joblib")
            tfidf_matrix_path = os.path.join("artifacts", "tfidf_matrix.joblib")
            movie_mapping_path = os.path.join("artifacts", "movie_mapping.joblib")
            
            # Load matrix files and dataframes from disk memory
            self.movies_df = joblib.load(processed_movies_path)
            self.movie_features_svd = joblib.load(movie_features_svd_path)
            self.tfidf_matrix = joblib.load(tfidf_matrix_path)
            self.movie_to_idx = joblib.load(movie_mapping_path)
            
            # SAFETY CHECK: Convert movie_to_idx securely regardless of whether it was saved 
            # as a Dict, a Pandas Series, or a Pandas Index structure.
            if hasattr(self.movie_to_idx, 'to_dict'):
                self.movie_dict = self.movie_to_idx.to_dict()
            elif isinstance(self.movie_to_idx, (pd.Index, pd.Series)):
                self.movie_dict = {val: idx for idx, val in enumerate(self.movie_to_idx)}
            else:
                self.movie_dict = dict(self.movie_to_idx)
            
            # Create a standardized, case-insensitive mapping dictionary to prevent spelling/casing mismatches
            self.normalized_movie_dict = {str(k).lower().strip(): v for k, v in self.movie_dict.items()}
            
            # Generate the true reverse lookup dictionary mapping numeric index array positions back to string titles
            self.idx_to_movie = {v: k for k, v in self.movie_dict.items()}
            
            # Pre-compute and cache the TF-IDF titles index mapping to accelerate matrix alignment computations
            if 'title' in self.movies_df.columns:
                self._title_to_tfidf_cache = {str(t).lower().strip(): idx for idx, t in enumerate(self.movies_df['title'])}
            
            print("ML inference layers initialized from cached storage boundaries.")
            logging.info("ML inference layers successfully loaded with internal alignment guards.")
            
        except Exception as e:
            logging.error("Critical breakdown occurred during initialization of the PredictPipeline components.")
            raise CustomException(e, sys)

    def get_hybrid_recommendations(self, movie_title: str, alpha: float = 0.5, top_n: int = 5):
        """
        Blends collaborative scores and text similarity vectors linearly to yield recommendations.
        Features automatic index space mapping to align collaborative and text models perfectly.
        """
        try:
            # INPUT CLEANING: Strip white spaces and enforce lowercasing for search safety
            cleaned_title = str(movie_title).lower().strip()
            
            # 1. Confirm the movie exists within our collaborative catalog universe using the normalized cache
            if cleaned_title not in self.normalized_movie_dict:
                logging.warning(f"Query target '{movie_title}' not tracked in matrix parameters.")
                return None

            # 2. Extract internal collaborative positional vector row location safely
            movie_idx = self.normalized_movie_dict[cleaned_title]
            
            # Compute Collaborative Scoring (16,116 values)
            target_svd_vector = self.movie_features_svd[movie_idx].reshape(1, -1)
            coll_scores = cosine_similarity(target_svd_vector, self.movie_features_svd).flatten()
            
            # 3. Dynamic Alignment: Find the true TF-IDF row index for the query movie
            target_tfidf_idx = movie_idx # Fallback default
            
            if len(self.movies_df) == self.tfidf_matrix.shape[0] and 'title' in self.movies_df.columns:
                if cleaned_title in self._title_to_tfidf_cache:
                    target_tfidf_idx = self._title_to_tfidf_cache[cleaned_title]
            elif len(self.movies_df) == len(coll_scores):
                if 'tfidf_idx' in self.movies_df.columns:
                    target_tfidf_idx = int(self.movies_df.loc[movie_idx, 'tfidf_idx'])
                elif 'index' in self.movies_df.columns:
                    target_tfidf_idx = int(self.movies_df.loc[movie_idx, 'index'])

            # BOUNDARY GUARD: Prevent unexpected index out of bounds crashes during lookup mapping operations
            if target_tfidf_idx >= self.tfidf_matrix.shape[0]:
                logging.error(f"Index out of bounds mapping error: target_tfidf_idx {target_tfidf_idx} exceeds matrix shape.")
                target_tfidf_idx = 0 # Safe fallback target assignment to preserve site runtime stability

            # Compute raw text similarity scores across all items (e.g., 86,537 items)
            target_tfidf_vector = self.tfidf_matrix[target_tfidf_idx]
            raw_content_scores = linear_kernel(target_tfidf_vector, self.tfidf_matrix).flatten()
            
            # 4. Map the raw content scores into an array aligned with the collaborative catalog items (e.g., 16,116 items)
            aligned_content_scores = np.zeros_like(coll_scores)
            
            if len(self.movies_df) == self.tfidf_matrix.shape[0] and 'title' in self.movies_df.columns:
                for i in range(len(coll_scores)):
                    if i in self.idx_to_movie:
                        current_title = str(self.idx_to_movie[i]).lower().strip()
                        if current_title in self._title_to_tfidf_cache:
                            tfidf_lookup_idx = self._title_to_tfidf_cache[current_title]
                            if tfidf_lookup_idx < len(raw_content_scores):
                                aligned_content_scores[i] = raw_content_scores[tfidf_lookup_idx]
                        
            elif len(self.movies_df) == len(coll_scores):
                for i in range(len(coll_scores)):
                    orig_idx = i
                    if 'tfidf_idx' in self.movies_df.columns:
                        orig_idx = int(self.movies_df.loc[i, 'tfidf_idx'])
                    elif 'index' in self.movies_df.columns:
                        orig_idx = int(self.movies_df.loc[i, 'index'])
                    
                    # BOUNDARY GUARD: Verify index is within matrix bounds before indexing
                    if 0 <= orig_idx < len(raw_content_scores):
                        aligned_content_scores[i] = raw_content_scores[orig_idx]
            else:
                # Safe structural cutoff fallback if metadata indexes are missing entirely
                min_len = min(len(coll_scores), len(raw_content_scores))
                aligned_content_scores[:min_len] = raw_content_scores[:min_len]
            
            # 5. Execute Linear Combination Blending using perfectly aligned vectors
            hybrid_scores = (alpha * coll_scores) + ((1 - alpha) * aligned_content_scores)
            
            # 6. Extract high-to-low sorted array coordinates
            sorted_indices = np.argsort(hybrid_scores)[::-1]
            
            recommendations = []
            rank = 1
            
            for idx in sorted_indices:
                # SELF-FILTER SHIELD: Prevent the engine from recommending the input movie back to the user
                if idx == movie_idx:
                    continue
                
                # Verify that the array position has a corresponding valid movie title key mapped inside the lookup dict
                if idx not in self.idx_to_movie:
                    continue
                    
                recommended_title = self.idx_to_movie[idx]
                
                # Format scores out of 100% cleanly for delivery to web page template cards
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