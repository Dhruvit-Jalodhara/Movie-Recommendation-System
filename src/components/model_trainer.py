import os
import sys
import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import mlflow
import mlflow.sklearn
from urllib.parse import urlparse
from dataclasses import dataclass

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, load_object, calculate_batch_mse

@dataclass
class ModelTrainerConfig:
    """
    Configuration dataclass that designates the exact paths for storing the final trained 
    models and low-dimensional mathematical feature representations. Saving these vector spaces 
    directly prevents out-of-memory errors during live system deployment.
    """
    svd_model_path: str = os.path.join("artifacts", "svd_model.joblib")
    tfidf_model_path: str = os.path.join("artifacts", "tfidf_model.joblib")
    movie_features_svd_path: str = os.path.join("artifacts", "movie_features_svd.joblib")
    tfidf_matrix_path: str = os.path.join("artifacts", "tfidf_matrix.joblib")
    movie_mapping_path: str = os.path.join("artifacts", "movie_mapping.joblib")

class ModelTrainer:
    """
    Production training component that handles Collaborative Filtering (TruncatedSVD) 
    and Content-Based (TF-IDF Vectorizer) engines. It enforces memory-efficient sparse layouts,
    calculates system diagnostics, and mirrors experiment records with DagsHub MLflow tracking.
    """
    def __init__(self):
        """Initializes the configurations and sets the direct tracking endpoint for the DagsHub MLflow Server."""

        self.model_trainer_config = ModelTrainerConfig()

        # Direct URL targeting your live remote DagsHub MLflow logging registry
        self.tracking_uri = "https://dagshub.com/Dhruvit-Jalodhara/Movie-Recommendation-System.mlflow"
        
    def initiate_model_training(self, processed_movies_path: str, processed_pivot_path: str):
        """
        Ingests pre-compressed payloads, structures inference dictionary trackers, executes 
        matrix factorization and term-frequency models, evaluates scores, and uploads metrics to MLflow.
        
        Args:
            processed_movies_path (str): File location of the cleaned movies metadata dataframe.
            processed_pivot_path (str): File location of the compressed CSR matrix payload dictionary.
            
        Raises:
            CustomException: Wraps and logs any underlying failure within the execution blocks.
        """

        logging.info("Starting Model Training and MLflow Tracking Component...")
        try:
            # 1. LOAD COMPRESSED PIPELINE DATA 
            logging.info("Loading preprocessed joblib artifacts from disk...")
            movies_df = load_object(processed_movies_path)
            sparse_payload = load_object(processed_pivot_path)
            
            # Extract elements from the lightweight dictionary structure
            ratings_csr = sparse_payload['csr_matrix']  # The mathematical sparse representation tracking non-zero entries
            matrix_movie_ids = sparse_payload['movie_ids']  # Ordered index tracking rows to movie IDs
            
            # 2. GENERATE INFERENCE LOOKUP REFERENCE MAPS 
            logging.info("Generating mapping series matching human-readable titles to internal coordinate ranks...")
            
            # We align the internal matrix order with the master movie details via a clean structural merge
            mapping_df = pd.merge(pd.DataFrame({'movieId': matrix_movie_ids}), movies_df, on='movieId').reset_index()

            # Fast index lookups maps string movie titles directly to coordinate array row indices
            movie_to_idx = pd.Series(mapping_df['index'].values, index=mapping_df['title'])
            
            # --- 3. CONFIGURE MLFLOW TRACKING PARAMETERS ---
            logging.info("Connecting to the remote DagsHub MLflow Tracking Server...")
            mlflow.set_tracking_uri(self.tracking_uri)
            tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme
            
            # Asserts the specific experiment project lane inside the DagsHub portal dashboard
            mlflow.set_experiment("Hybrid_Recommender_Experiment")
            
            # Open the secure contextual session container to push experimental metrics
            with mlflow.start_run():
                logging.info("MLflow logging session successfully initialized.")
                
                # Baseline architectural parameters specified in your training scripts
                n_components = 50
                random_state = 42
                
                # 4. EXECUTE COLLABORATIVE FILTERING ENGINE (TruncatedSVD) 
                logging.info(f"Fitting TruncatedSVD matrix factorization with components={n_components}...")

                svd = TruncatedSVD(n_components=n_components, random_state=random_state)

                # Compresses the user-interaction vector spaces down to a dense latent representation matrix
                movie_features_svd = svd.fit_transform(ratings_csr)
                
                # Calculate Reconstruction Error using the specialized low-memory chunk processing algorithm
                logging.info("Running chunked batch evaluation for matrix reconstruction MSE score...")
                svd_mse = calculate_batch_mse(movie_features_svd, svd.components_, ratings_csr)
                
                # Calculate System Item Catalog Coverage using a deterministic 200 item evaluation framework
                logging.info("Calculating unique catalog coverage metrics across test slice...")
                np.random.seed(random_state)
                total_unique_movies = ratings_csr.shape[0]
                sample_indices = np.random.choice(total_unique_movies, size=200, replace=False)
                
                # Temporary similarity evaluation block for validation metrics calculations
                svd_sim_matrix = cosine_similarity(movie_features_svd)
                svd_recommended_set = set()
                
                for idx in sample_indices:
                    svd_scores = svd_sim_matrix[idx]
                    # Isolate the top 10 closely matching index entities, skipping the targeted track entry itself
                    top_svd = np.argsort(svd_scores)[::-1][1:11]
                    svd_recommended_set.update(top_svd)
                    
                # Quantifies the overall percentage of the catalogue the SVD system can successfully surface
                svd_coverage = (len(svd_recommended_set) / total_unique_movies) * 100
                
                # 5. EXECUTE TEXT NLP FILTERING ENGINE (TF-IDF Vectorizer)
                logging.info("Fitting TF-IDF Vectorizer engine onto metadata text features...")
                tfidf = TfidfVectorizer(stop_words='english')

                # Computes unique multi-word token weighting matrices for the unified narrative text blocks
                tfidf_matrix = tfidf.fit_transform(movies_df['metadata_soup'])
                
                # 6. SYNCHRONIZE METRICS AND SETTINGS TO REMOTE DASHBOARD 
                logging.info("Pushing operational configurations and metrics to DagsHub MLflow Server...")

                # Parameters (System Constants)
                mlflow.log_param("svd_n_components", n_components)
                mlflow.log_param("svd_random_state", random_state)
                mlflow.log_param("tfidf_stop_words", "english")
                
                # Metrics (Dynamic Performance Outputs)
                mlflow.log_metric("svd_reconstruction_mse", svd_mse)
                mlflow.log_metric("svd_catalog_coverage_pct", svd_coverage)
                
                # 7. PERSIST TRAINED ARTIFACTS AND REGISTRIES 
                logging.info("Saving core processing structures locally into artifacts/ space...")
                save_object(self.model_trainer_config.svd_model_path, svd)
                save_object(self.model_trainer_config.tfidf_model_path, tfidf)
                save_object(self.model_trainer_config.movie_features_svd_path, movie_features_svd)
                save_object(self.model_trainer_config.tfidf_matrix_path, tfidf_matrix)
                save_object(self.model_trainer_config.movie_mapping_path, movie_to_idx)
                
                # Verifies backend storage schema before registering binaries into the remote DagsHub Model Registry
                if tracking_url_type_store != "file":
                    logging.info("Registering trained modules directly into the cloud Model Hub...")
                    mlflow.sklearn.log_model(svd, "svd_model", registered_model_name="SVD_Collaborative_Engine")
                    mlflow.sklearn.log_model(tfidf, "tfidf_model", registered_model_name="TFIDF_Content_Engine")
                else:
                    mlflow.sklearn.log_model(svd, "svd_model")
                    mlflow.sklearn.log_model(tfidf, "tfidf_model")
                    
            logging.info("Model Training and remote MLflow tracking completed successfully!")
            
        except Exception as e:
            logging.error("Exception encountered during Model Training phase.")
            raise CustomException(e, sys)