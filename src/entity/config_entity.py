from datetime import datetime
import os

from src.constants.constant import *
from dataclasses import dataclass



TIMESTAMP: str = datetime.now().strftime("%m_%d_%Y_%H_%M_%S")

@dataclass
class TrainingPipelineConfig:
    pipeline_name: str = PIPELINE_NAME
    artifact_dir: str = os.path.join(ARTIFACT_DIR, TIMESTAMP)
    timestamp: str = TIMESTAMP


training_pipeline_config: TrainingPipelineConfig = TrainingPipelineConfig()

@dataclass
class DataIngestionConfig:
    data_ingestion_dir: str = os.path.join(training_pipeline_config.artifact_dir, DATA_INGESTION_DIR_NAME)
    feature_store_file_path: str = os.path.join(data_ingestion_dir, DATA_INGESTION_FEATURE_STORE_DIR, FILE_NAME)
    training_file_path: str = os.path.join(data_ingestion_dir, DATA_INGESTION_INGESTED_DIR, TRAIN_FILE_NAME)
    testing_file_path: str = os.path.join(data_ingestion_dir, DATA_INGESTION_INGESTED_DIR, TEST_FILE_NAME)
    train_test_split_ratio: float = DATA_INGESTION_TRAIN_TEST_SPLIT_RATIO
    collection_name:str = DATA_INGESTION_COLLECTION_NAME


@dataclass
class DataValidationConfig:
    data_validation_dir: str = os.path.join(training_pipeline_config.artifact_dir, DATA_VALIDATION_DIR_NAME)
    validation_report_file_path: str = os.path.join(data_validation_dir, DATA_VALIDATION_REPORT_FILE_NAME)


class DataTransformationConfig:
    def __init__(self, training_pipeline_config: TrainingPipelineConfig):
        self.data_transformation_dir: str = os.path.join(
            training_pipeline_config.artifact_dir, constant.DATA_TRANSFORMATION_DIR_NAME
        )
        self.transformed_train_file_path: str = os.path.join(
            self.data_transformation_dir,
            constant.DATA_TRANSFORMATION_TRANSFORMED_DATA_DIR,
            constant.TRAIN_FILE_NAME.replace("csv", "npy"),
        )
        self.transformed_test_file_path: str = os.path.join(
            self.data_transformation_dir,
            constant.DATA_TRANSFORMATION_TRANSFORMED_DATA_DIR,
            constant.TEST_FILE_NAME.replace("csv", "npy"),
        )
        self.transformed_object_file_path: str = os.path.join(
            self.data_transformation_dir,
            constant.DATA_TRANSFORMATION_TRANSFORMED_OBJECT_DIR,
            constant.PREPROCSSING_OBJECT_FILE_NAME,
        )

class ModelTrainingConfig:
    def __init__(self, training_pipeline_config: TrainingPipelineConfig):
        self.model_trainer_dir : str = os.path.join(training_pipeline_config.artifact_dir, constant.MODEL_TRAINER_DIR_NAME)
        self.trained_model_file_path : str = os.path.join(self.model_trainer_dir, constant.MODEL_TRAINER_TRAINED_MODEL_DIR, constant.MODEL_FILE_NAME)
        self.expected_accuracy : float = constant.MODEL_TRAINER_EXPECTED_SCORE
        self.overfitting_underfitting_threshold : float = constant.MODEL_TRAINER_OVERFITTING_UNDERFITTING_THRESHOLD 
        

class ModelStorageConfig:
    def __init__(self):
        self.local_model_dir = os.path.join(os.getcwd(), "saved_models")
        self.local_model_path = os.path.join(self.local_model_dir, "trained_model.pkl")
        os.makedirs(self.local_model_dir, exist_ok=True)