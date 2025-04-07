from src.exception.exception import VehicleInsuranceException
from src.logging.logger import logging

from src.constants import constant
from src.entity.config_entity import DataIngestionConfig
from src.entity.artifact_entity import DataIngestionArtifact

import os
import sys
import pymongo
import numpy as np
from pandas import DataFrame
from typing import List
from sklearn.model_selection import train_test_split
from src.data_access.fetch_data import FetchData

from dotenv import load_dotenv

load_dotenv()

MONGO_DB_URL = os.getenv("MONGO_DB_URL")


class DataIngestion:
    def __init__(self, data_ingestion_config: DataIngestionConfig = DataIngestionConfig):
        """
        param data_ingestion_config: configuration for data ingestion

        """
        try:
            self.data_ingestion_config = data_ingestion_config
            # self.mongo_client = pymongo.MongoClient(MONGO_DB_URL)
        except Exception as e:
            raise VehicleInsuranceException(e, sys)

    def export_data_into_feature_store(self) -> DataFrame:
        """
        Method Name :   export_data_into_feature_store
        Description :   This method exports data from mongodb to csv file

        Output      :   data is returned as artifact of data ingestion components
        On Failure  :   Write an exception log and then raise an exception
        """
        try:
            logging.info("Exporting data from MongoDB")
            data = FetchData()
            dataframe = data.export_collection_as_dataframe(
                collection_name=self.data_ingestion_config.collection_name
            )
            logging.info(f"Shape of dataframe: {dataframe.shape}")
            feature_store_file_path = self.data_ingestion_config.feature_store_file_path
            dir_path = os.path.dirname(feature_store_file_path)
            os.makedirs(dir_path, exist_ok=True)
            logging.info(
                f"Saving exported data into feature store file path: {feature_store_file_path}"
            )
            dataframe.to_csv(feature_store_file_path, index=False, header=True)
            return dataframe
        except Exception as e:
            raise VehicleInsuranceException(e, sys)

    def split_data_as_train_test(self, dataframe: DataFrame) -> None:
        """
        Method Name :   split_data_into_train_test
        Description :   This method splits the dataframe into train set and test set based on split ratio

        Output      :   Train and test datasets are saved to CSV files
        On Failure  :   Writes an exception log and raises an exception
        """
        logging.info("Entered split_data_into_train_test method of DataIngestion class")
        try:
            train_set, test_set = train_test_split(dataframe, test_size=self.data_ingestion_config.train_test_split_ratio)
            logging.info("Performed train-test split on the dataframe")

            dir_path = os.path.dirname(self.data_ingestion_config.training_file_path)
            os.makedirs(dir_path, exist_ok=True)

            logging.info("Exporting train and test datasets to CSV files.")

            train_set.to_csv(self.data_ingestion_config.training_file_path, index=False, header=True)
            test_set.to_csv(self.data_ingestion_config.testing_file_path, index=False, header=True)

            logging.info("Exporting completed successfully.")
        except Exception as e:
            raise VehicleInsuranceException(e, sys)

    def initiate_data_ingestion(self) -> DataIngestionArtifact:
        """
        Method Name :   initiate_data_ingestion
        Description :   This method initiates the data ingestion process

        Output      :   Returns the DataIngestionArtifact containing train and test file paths
        On Failure  :   Writes an exception log and raises an exception
        """
        logging.info("Entered initiate_data_ingestion method of DataIngestion class")
        try:
            dataframe = self.export_data_into_feature_store()

            logging.info("Successfully fetched data from MongoDB")

            self.split_data_as_train_test(dataframe)

            logging.info("Performed train-test split on the dataset")

            logging.info("Exited initiate_data_ingestion method of DataIngestion class")

            data_ingestion_artifact = DataIngestionArtifact(
                trained_file_path=self.data_ingestion_config.training_file_path,
                test_file_path=self.data_ingestion_config.testing_file_path,
            )
            logging.info(f"Data ingestion artifact created: {data_ingestion_artifact}")

            return data_ingestion_artifact
        except Exception as e:
            raise VehicleInsuranceException(e, sys)


# below is the code to test the data ingestion component
# from src.components.data_ingestion import DataIngestion
# from src.exception.exception import VehicleInsuranceException
# from src.logging.logger import logging
# from src.entity.config_entity import DataIngestionConfig
# from src.entity.config_entity import TrainingPipelineConfig
# import sys

# if __name__ == "__main__":
#    try:
#        logging.info("Data Ingestion Started")
#        traning_pipeline_config = TrainingPipelineConfig()
#        data_ingestion_config = DataIngestionConfig(traning_pipeline_config)
#        data_ingestion = DataIngestion(data_ingestion_config)
#        data_ingestion_artifact = data_ingestion.initiate_data_ingestion()
#       print(data_ingestion_artifact)
#   except Exception as e:
#       raise VehicleInsuranceException(e, sys)
