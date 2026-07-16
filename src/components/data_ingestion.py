import os
import sys
from src.logger import logging
from src.exception import CustomeExpection
from dataclasses import dataclass
import pandas as pd
from sklearn.model_selection import train_test_split


@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join('artifacts', 'train.csv')
    test_data_path: str = os.path.join('artifacts', 'test.csv')
    raw_data_path: str = os.path.join('artifacts', 'raw_data.csv')


class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info('Data ingestion has start')
        try:
            logging.info('Read the dataset')
            df = pd.read_csv("./Notebook/job_title_data.csv")
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)

            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)

            logging.info('Train test split the data')
            train_set, test_set = train_test_split(df, random_state=42, test_size=0.2)

            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)

            logging.info('The data ingestion is complete')

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )

        except Exception as e:
            raise CustomeExpection(e, sys)

    