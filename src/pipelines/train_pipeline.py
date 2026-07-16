import sys

from src.logger import logging
from src.exception import CustomeExpection

from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer


class TrainingPipeline:
    def __init__(self):
        self.data_ingestion = DataIngestion()
        self.data_transformation = DataTransformation()
        self.model_trainer = ModelTrainer()

    def run_pipeline(self):
        try:
            logging.info('Training pipeline started ')

            logging.info(' Data ingestion')
            train_data_path, test_data_path = self.data_ingestion.initiate_data_ingestion()

            logging.info('Data transformation')
            train_arr, test_arr, preprocessor_path = self.data_transformation.InitiateDataTransformation(
                train_data_path, test_data_path
            )

            logging.info(' Model training')
            best_model_name, best_model_score = self.model_trainer.initiate_model_trainer(
                train_arr, test_arr, preprocessor_path
            )

            logging.info('Training pipeline completed successfully')
            print(f"Best model: {best_model_name} | Test accuracy: {best_model_score:.4f}")

            return best_model_name, best_model_score

        except Exception as e:
            raise CustomeExpection(e, sys)


if __name__ == '__main__':
    pipeline = TrainingPipeline()
    pipeline.run_pipeline()