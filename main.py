from src.components.data_ingestion import DataIngestion
from src.components.data_validation import DataValidation

from src.exception.exception import VehicleInsuranceException
from src.logging.logger import logging
from src.entity.config_entity import DataIngestionConfig, DataValidationConfig
from src.entity.config_entity import TrainingPipelineConfig
import sys

if __name__ == "__main__":
    try:
        traning_pipeline_config = TrainingPipelineConfig()
        data_ingestion_config = DataIngestionConfig(traning_pipeline_config)
        logging.info("Data Ingestion Started")
        data_ingestion = DataIngestion(data_ingestion_config)
        data_ingestion_artifact = data_ingestion.initiate_data_ingestion()
        logging.info("Data Ingestion completed")
        print(data_ingestion_artifact)

        data_validation_config = DataValidationConfig(traning_pipeline_config)
        logging.info("Data Validation Started")
        data_validation = DataValidation(
            data_ingestion_artifact, data_validation_config
        )
        data_validation_artifact = data_validation.initiate_data_validation()
        logging.info("Data Validation completed")
        print(data_validation_artifact)
    except Exception as e:
        raise VehicleInsuranceException(e, sys)
