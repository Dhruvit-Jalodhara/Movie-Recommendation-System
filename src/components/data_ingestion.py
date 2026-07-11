import os
import sys
from dataclasses import dataclass

from src.exception import CustomException
from src.logger import logging

@dataclass
class DataIngestionConfig:
    """
    Configuration data class defining the absolute file paths for the raw, DVC-tracked dataset files.
    These paths point directly to the local storage to prevent redundant artifact duplication in memory.
    """
    raw_movies_path: str = "/Users/dhruvitjalodhara/programming/ML Practice/Movie Recommendation System/movie dataset/raw/movies.csv"
    raw_ratings_path: str = "/Users/dhruvitjalodhara/programming/ML Practice/Movie Recommendation System/movie dataset/raw/ratings.csv"
    raw_tags_path: str = "/Users/dhruvitjalodhara/programming/ML Practice/Movie Recommendation System/movie dataset/raw/tags.csv"

class DataIngestion:
    """
    Component responsible for validating the presence of raw data files managed by DVC.
    It acts as a secure gateway to ensure data integrity before the heavy preprocessing phase begins.
    """
    def __init__(self):
        """Initializes the DataIngestion component by loading the required file path configurations."""
        self.ingestion_config = DataIngestionConfig()
        
    def initiate_data_ingestion(self):
        """
        Validates the existence of the raw movies, ratings, and tags CSV files on the local machine.
        
        Returns:
            tuple: A tuple containing the validated absolute file paths for (movies, ratings, tags).
            
        Raises:
            CustomException: If any of the required raw data files are missing from the local directory.
        """
        logging.info("Starting Data Ingestion phase...")
        try:
            logging.info("Validating local DVC-tracked raw datasets...")
            
            # Defensive check: Ensure the files exist before passing them down the pipeline
            if not os.path.exists(self.ingestion_config.raw_movies_path):
                raise Exception(f"Missing file: {self.ingestion_config.raw_movies_path}. Did you forget to run 'dvc pull'?")
                
            if not os.path.exists(self.ingestion_config.raw_ratings_path):
                raise Exception(f"Missing file: {self.ingestion_config.raw_ratings_path}. Did you forget to run 'dvc pull'?")
                
            if not os.path.exists(self.ingestion_config.raw_tags_path):
                raise Exception(f"Missing file: {self.ingestion_config.raw_tags_path}. Did you forget to run 'dvc pull'?")
            
            logging.info("Data validation completed! Bypassing artifact duplication.")
            
            # Pass the direct DVC paths straight into the preprocessing component
            return (
                self.ingestion_config.raw_movies_path,
                self.ingestion_config.raw_ratings_path,
                self.ingestion_config.raw_tags_path
            )
            
        except Exception as e:
            logging.error("Exception encountered during Ingestion lifecycle.")
            raise CustomException(e, sys)