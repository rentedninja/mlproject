import os
import sys
import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from src.exception import CustomException

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

@dataclass
class DataIngestionConfig:
    base_path: str = os.path.abspath(os.path.dirname(__file__))  # Absolute path to script's directory
    train_data_path: str = os.path.join(base_path, 'artifacts', "train.csv")
    test_data_path: str = os.path.join(base_path, 'artifacts', "test.csv")
    raw_data_path: str = os.path.join(base_path, 'artifacts', "data.csv")

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
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)

            # Save the raw data
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)
            logging.info(f"Raw data saved at: {self.ingestion_config.raw_data_path}")

            # Perform train-test split
            logging.info("Performing train-test split...")
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)

            # Save the train and test datasets
            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)

            logging.info(f"Train data saved at: {self.ingestion_config.train_data_path}")
            logging.info(f"Test data saved at: {self.ingestion_config.test_data_path}")

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

        logging.info(f"Train data path: {train_data}")
        logging.info(f"Test data path: {test_data}")

    except CustomException as ce:
        logging.error(f"Custom exception occurred: {str(ce)}")
    except Exception as e:
        logging.error(f"An unexpected error occurred: {str(e)}")
