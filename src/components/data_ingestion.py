# src/components/data_ingestion.py

import os
import sys
from src.exception import CustomException  # Import CustomException from src
from src.logger import logging  # Import logging from src
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from src.components.data_transformation import DataTransformation  # Importing DataTransformation from components
from src.components.data_transformation import DataTransformationConfig  # Import DataTransformationConfig from components
from src.components.model_trainer import ModelTrainerConfig  # Import ModelTrainerConfig
from src.components.model_trainer import ModelTrainer  # Import ModelTrainer

@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join('artifacts', "train.csv")
    test_data_path: str = os.path.join('artifacts', "test.csv")
    raw_data_path: str = os.path.join('artifacts', "data.csv")

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion method or component")
        try:
            # Adjust path if necessary
            df = pd.read_csv(r'E:\Code\Mlproject\notebook\data\stud.csv')  # Ensure path is correct
            logging.info('Read the dataset as dataframe')

            # Create the directory if it does not exist
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)

            # Save raw data
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)

            logging.info("Train test split initiated")
            # Train test split
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)

            # Save train and test data
            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)

            logging.info("Ingestion of the data is completed")

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
        except Exception as e:
            raise CustomException(e, sys)

if __name__ == "__main__":
    # Ensure to run this script from E:\Code\Mlproject directory
    obj = DataIngestion()
    train_data, test_data = obj.initiate_data_ingestion()

    # Data transformation
    data_transformation = DataTransformation()
    train_arr, test_arr, _ = data_transformation.initiate_data_transformation(train_data, test_data)

    # Model trainer
    modeltrainer = ModelTrainer()
    print(modeltrainer.initiate_model_trainer(train_arr, test_arr))