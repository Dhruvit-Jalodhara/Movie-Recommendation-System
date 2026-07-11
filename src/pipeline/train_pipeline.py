import sys
from src.logger import logging
from src.exception import CustomException
from src.components.data_ingestion import DataIngestion
from src.components.data_preprocessing import DataPreprocessing
from src.components.model_trainer import ModelTrainer

class TrainPipeline:
    """
    Orchestration pipeline responsible for executing the end-to-end model training lifecycle.
    It coordinates data ingestion validation, structural preprocessing, and MLflow-tracked model training.
    """

    def __init__(self):
        """Initializes the TrainPipeline orchestrator."""
        pass
        
    def run_pipeline(self):
        """
        Sequentially runs all machine learning lifecycle stages.
        Passes data artifacts down the pipeline stream while logging execution steps.
        
        Raises:
            CustomException: If any nested component fails during execution.
        """
        logging.info("Training Pipeline execution started!")
        try:
            # Step 1: Data Ingestion Validation
            logging.info("--> Initiating Data Ingestion Component")
            ingestion = DataIngestion()
            raw_movies_path, raw_ratings_path, raw_tags_path = ingestion.initiate_data_ingestion()
            
            # Step 2: Data Preprocessing & Sparsity Filtering
            logging.info("--> Initiating Data Preprocessing Component")
            preprocessor = DataPreprocessing()
            processed_movies_path, processed_pivot_path = preprocessor.initiate_data_preprocessing(
                movies_path=raw_movies_path, 
                ratings_path=raw_ratings_path, 
                tags_path=raw_tags_path
            )
            
            # Step 3: Model Training & MLflow Synchronization
            logging.info("--> Initiating Model Trainer Component")
            trainer = ModelTrainer()
            trainer.initiate_model_training(
                processed_movies_path=processed_movies_path, 
                processed_pivot_path=processed_pivot_path
            )
            
            logging.info("Training Pipeline execution completed successfully!")
            
        except Exception as e:
            logging.error("Critical breakdown occurred within the Training Pipeline execution block.")
            raise CustomException(e, sys)

# The single entry point to run the entire training process
if __name__ == "__main__":
    pipeline = TrainPipeline()
    pipeline.run_pipeline()