from src.components.data_ingestion import DataIngestion
from src.components.data_validation import DataValidation
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer

from src.exception.exception import VehicleInsuranceException
from src.logging.logger import logging
from src.entity.config_entity import (
    DataIngestionConfig,
    DataValidationConfig,
    DataTransformationConfig,
    ModelTrainingConfig
)
from src.entity.config_entity import TrainingPipelineConfig
import sys


if __name__ == "__main__":
    try:
        traning_pipeline_config = TrainingPipelineConfig()
        
        # Data Ingestion
        data_ingestion_config = DataIngestionConfig(traning_pipeline_config)
        logging.info("Data Ingestion Started")
        data_ingestion = DataIngestion(data_ingestion_config)
        data_ingestion_artifact = data_ingestion.initiate_data_ingestion()
        logging.info("Data Ingestion completed")
        print(data_ingestion_artifact)

        # Data Validation
        data_validation_config = DataValidationConfig(traning_pipeline_config)
        logging.info("Data Validation Started")
        data_validation = DataValidation(
            data_ingestion_artifact, data_validation_config
        )
        data_validation_artifact = data_validation.initiate_data_validation()
        logging.info("Data Validation completed")
        print(data_validation_artifact)

        # Data Transformation
        datatransformationconfig = DataTransformationConfig(traning_pipeline_config)
        logging.info(f"Data transformation started")
        data_transformation = DataTransformation(
            data_ingestion_artifact, datatransformationconfig, data_validation_artifact
        )
        data_transformation_artifact = (
            data_transformation.initiate_data_transformation()
        )
        logging.info("Data transformation completed")
        print(data_transformation_artifact)
        
        ## Model Training
        model_training_config = ModelTrainingConfig(traning_pipeline_config)
        logging.info("Model training started")
        model_trainer=ModelTrainer(model_trainer_config=model_training_config, data_transformation_artifact=data_transformation_artifact)
        model_trainer_artifact = model_trainer.initiate_model_trainer()
        logging.info("Model training completed.")

    except Exception as e:
        raise VehicleInsuranceException(e, sys)
