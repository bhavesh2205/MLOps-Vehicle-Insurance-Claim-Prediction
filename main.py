from src.components.data_ingestion import DataIngestion
from src.exception.exception import VehicleInsuranceException
from src.logging.logger import logging
from src.entity.config_entity import DataIngestionConfig
from src.entity.config_entity import TrainingPipelineConfig
import sys

if __name__ == "__main__":
    try:
        logging.info("Data Ingestion Started")
        traning_pipeline_config = TrainingPipelineConfig()
        data_ingestion_config = DataIngestionConfig(traning_pipeline_config)
        data_ingestion = DataIngestion(data_ingestion_config)
        data_ingestion_artifact = data_ingestion.initiate_data_ingestion()
        print(data_ingestion_artifact)
    except Exception as e:
        raise VehicleInsuranceException(e, sys)
