from src.exception.exception import VehicleInsuranceException
from src.logging.logger import logging

from src.utils.main_utils import read_yaml_file
from src.entity.artifact_entity import DataIngestionArtifact, DataValidationArtifact
from src.entity.config_entity import DataValidationConfig
from src.constants.constant import SCHEMA_FILE_PATH
import json

import sys, os
import pandas as pd
from pandas import DataFrame


class DataValidation:
    def __init__(self,data_ingestion_artifact: DataIngestionArtifact,data_validation_config: DataValidationConfig):
        """
        param data_ingestion_artifact: Output reference of data ingestion artifact stage
        param data_validation_config: configuration for data validation
        """
        try:
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_validation_config = data_validation_config
            self._schema_config = read_yaml_file(file_path=SCHEMA_FILE_PATH)
        except Exception as e:
            raise VehicleInsuranceException(e, sys)

    def validate_number_of_columns(self, dataframe: DataFrame) -> bool:
        """
        param dataframe: Input DataFrame
        return: bool
        """
        try:
            number_of_columns = len(self._schema_config["columns"])
            logging.info(f"Number of columns in the schema: {number_of_columns}")
            logging.info(f"Number of columns in the dataframe: {len(dataframe.columns)}")
            
            if len(dataframe.columns) == number_of_columns:
                return True
            else:
                return False
        except Exception as e:
            raise VehicleInsuranceException(e, sys)

    def is_column_exist(self, dataframe: pd.DataFrame) -> bool:
        """
        Method Name :   is_column_exist
        Description :   This method validates the existence of a numerical and categorical columns

        Output      :   Returns bool value based on validation results
        On Failure  :   Write an exception log and then raise an exception
        """
        try:
            dataframe_columns = dataframe.columns
            missing_numerical_columns = []
            missing_categorical_columns = []
            for column in self._schema_config["numerical_columns"]:
                if column not in dataframe_columns:
                    missing_numerical_columns.append(column)

            if len(missing_numerical_columns) > 0:
                logging.info(f"Missing numerical column: {missing_numerical_columns}")

            for column in self._schema_config["categorical_columns"]:
                if column not in dataframe_columns:
                    missing_categorical_columns.append(column)

            if len(missing_categorical_columns) > 0:
                logging.info(
                    f"Missing categorical column: {missing_categorical_columns}"
                )

            return False if len(missing_categorical_columns) > 0 or len(missing_numerical_columns) > 0 else True
        except Exception as e:
            raise VehicleInsuranceException(e, sys)

    @staticmethod
    def read_data(file_path) -> pd.DataFrame:
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise VehicleInsuranceException(e, sys)

    def initiate_data_validation(self) -> DataValidationArtifact:
        """
        Method Name :   initiate_data_validation
        Description :   This method initiates the data validation component for the pipeline

        Output      :   Returns bool value based on validation results
        On Failure  :   Write an exception log and then raise an exception
        """

        try:
            validation_error_message = ""
            logging.info("Starting data validation")
            train_df, test_df = (
                DataValidation.read_data(
                    file_path=self.data_ingestion_artifact.trained_file_path
                ),
                DataValidation.read_data(
                    file_path=self.data_ingestion_artifact.test_file_path
                ),
            )

            # check length of train and test dataframe
            status = self.validate_number_of_columns(dataframe=train_df)
            if not status:
                validation_error_message += f"Columns are missing in training dataframe."
            else:
                logging.info(
                    f"All required columns present in training dataframe: {status}"
                )

            status = self.validate_number_of_columns(dataframe=test_df)
            if not status:
                validation_error_message += f"Columns are missing in test dataframe. "
            else:
                logging.info(f"All required columns present in testing dataframe: {status}")

            # Validating col dtype for train/test df
            status = self.is_column_exist(dataframe=train_df)
            if not status:
                validation_error_message += f"Columns are missing in training dataframe. "
            else:
                logging.info(
                    f"All categorical/int columns present in training dataframe: {status}"
                )

            status = self.is_column_exist(dataframe=test_df)
            if not status:
                validation_error_message += f"Columns are missing in test dataframe."
            else:
                logging.info(
                    f"All categorical/numerical columns present in testing dataframe: {status}"
                )

            validation_status = len(validation_error_message) == 0

            data_validation_artifact = DataValidationArtifact(
                validation_status=validation_status,
                message=validation_error_message,
                validation_report_file_path=self.data_validation_config.validation_report_file_path,
            )

            # Ensure the directory for validation_report_file_path exists
            report_dir = os.path.dirname(self.data_validation_config.validation_report_file_path)
            os.makedirs(report_dir, exist_ok=True)

            # Save validation status and message to a JSON file
            validation_report = {
                "validation_status": validation_status,
                "message": validation_error_message.strip(),
            }

            with open(self.data_validation_config.validation_report_file_path, "w") as report_file:
                json.dump(validation_report, report_file, indent=4)

            logging.info("Data validation artifact created and saved to JSON file.")
            logging.info(f"Data validation artifact: {data_validation_artifact}")
            return data_validation_artifact
        except Exception as e:
            raise VehicleInsuranceException(e, sys) from e
