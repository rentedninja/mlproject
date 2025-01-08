import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd
import dill
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from src.components.data_transformation import DataTransformation
from src.components.data_transformation import DataTransformationConfig


from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainerConfig
from src.components.model_trainer import ModelTrainer

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

@dataclass
class DataIngestionConfig:
    # Use current working directory for placing artifacts folder outside src or components
    base_path: str = os.getcwd()
    artifacts_path: str = os.path.join(base_path, 'artifacts')
    train_data_path: str = os.path.join(artifacts_path, "train.csv")
    test_data_path: str = os.path.join(artifacts_path, "test.csv")
    raw_data_path: str = os.path.join(artifacts_path, "data.csv")

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion method or component")
        try:
            # Ensure the file exists before proceeding
            if not os.path.exists('notebook/data/StudentsPerformance.csv'):
                logging.error("Dataset file does not exist.")
                raise FileNotFoundError("Dataset file not found.")
            
            # Read the dataset
            df = pd.read_csv('notebook/data/StudentsPerformance.csv')
            logging.info('Dataset read successfully.')

            # Create the 'artifacts' directory if it doesn't exist
            os.makedirs(self.ingestion_config.artifacts_path, exist_ok=True)

            # Log absolute file paths
            raw_data_abs_path = os.path.abspath(self.ingestion_config.raw_data_path)
            logging.info(f"Saving raw data to: {raw_data_abs_path}")
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)
            logging.info(f"Raw data saved at: {raw_data_abs_path}")

            # Perform train-test split
            logging.info("Performing train-test split...")
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)

            # Log the absolute paths for train and test data
            train_data_abs_path = os.path.abspath(self.ingestion_config.train_data_path)
            test_data_abs_path = os.path.abspath(self.ingestion_config.test_data_path)

            # Save the train and test datasets
            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)

            logging.info(f"Train data saved at: {train_data_abs_path}")
            logging.info(f"Test data saved at: {test_data_abs_path}")

            logging.info("Data ingestion completed successfully.")

            return self.ingestion_config.train_data_path, self.ingestion_config.test_data_path

        except FileNotFoundError as fnf_error:
            logging.error(f"Error: {fnf_error}")
            raise CustomException(fnf_error, sys)

        except Exception as e:
            logging.error(f"An unexpected error occurred: {str(e)}")
            raise CustomException(e, sys)


if __name__ == "__main__":
    try:
        # Instantiate the DataIngestion class and call the ingestion method
        obj = DataIngestion()
        train_data, test_data = obj.initiate_data_ingestion()
        data_transformation = DataTransformation()
        train_arr, test_arr, _ = data_transformation.initiate_data_transformation(train_data, test_data)

        modeltrainer=ModelTrainer()
        modeltrainer.initiate_model_trainer(train_arr,test_arr)

        modeltrainer=ModelTrainer()
        print(modeltrainer.initiate_model_trainer(train_arr,test_arr))

        logging.info(f"Train data path: {train_data}")
        logging.info(f"Test data path: {test_data}")

    except CustomException as ce:
        logging.error(f"Custom exception occurred: {str(ce)}")
    except Exception as e:
        logging.error(f"An unexpected error occurred: {str(e)}")
