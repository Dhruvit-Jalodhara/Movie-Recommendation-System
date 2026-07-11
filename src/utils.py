import os
import sys
import joblib
import numpy as np

from src.exception import CustomException
from src.logger import logging

def save_object(file_path, obj):
    """Saves a Python object (model, matrix, preprocessor) to a specified file path."""
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            joblib.dump(obj, file_obj)

        logging.info(f"Successfully saved object at {file_path}")

    except Exception as e:
        raise CustomException(e, sys)

def load_object(file_path):
    """Loads a saved Python object from a specified file path."""
    try:
        if not os.path.exists(file_path):
            raise Exception(f"File path {file_path} does not exist.")
            
        with open(file_path, "rb") as file_obj:
            loaded_obj = joblib.load(file_obj)
            
        return loaded_obj

    except Exception as e:
        raise CustomException(e, sys)

def calculate_batch_mse(W, H, original_csr, batch_size=1000):
    """
    Calculates Mean Squared Error in small, memory-safe chunks.
    This metric will be logged directly to MLflow.

    parameters:
    ------------
    
    W: Movie features matrix (e.g., svd.fit_transform output)
    H: Components matrix (e.g., svd.components_)
    original_csr: Compressed sparse matrix (ratings_csr)
    """
    try:
        total_squared_error = 0.0
        num_movies = original_csr.shape[0]
        total_elements = original_csr.shape[0] * original_csr.shape[1]

        # Process the matrix in small chunks of rows
        for start_idx in range(0, num_movies, batch_size):
            end_idx = min(start_idx + batch_size, num_movies)

            # 1. Get the dense chunk of original ratings directly from the sparse matrix
            actual_chunk = original_csr[start_idx:end_idx].toarray()

            # 2. Reconstruct ONLY this small chunk of rows
            predicted_chunk = np.dot(W[start_idx:end_idx], H)

            # 3. Sum up the squared differences for this chunk
            total_squared_error += np.sum((actual_chunk - predicted_chunk) ** 2)

        # Final step: Divide the grand total by the total number of cells
        return total_squared_error / total_elements
        
    except Exception as e:
        raise CustomException(e, sys)