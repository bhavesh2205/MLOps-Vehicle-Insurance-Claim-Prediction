import os
import sys
from datetime import datetime, date
import pandas as pd
import numpy as np

from dotenv import load_dotenv
load_dotenv()

# For MongoDB connection
DATABASE_NAME = "vehicle"
COLLECTION_NAME = "vehicleInsurance"
MONGODB_URL_KEY = os.getenv("MONGODB_URL_KEY")

# Pipeline related constant start with PIPELINE
TARGET_COLUMN = "outcome"
PIPELINE_NAME: str = ""
ARTIFACT_DIR: str = "artifacts"
FILE_NAME: str = "insurance_data.csv"

TRAIN_FILE_NAME: str = "train.csv"
TEST_FILE_NAME: str = "test.csv"

# Model related constant start with MODEL
SCHEMA_FILE_PATH = os.path.join("schema", "schema.yaml")
MODEL_FILE_NAME = "model.pkl"
SAVED_MODEL_DIR = os.path.join("saved_models")
PREPROCSSING_OBJECT_FILE_NAME = "preprocessing.pkl"

# AWS related constant start with AWS
AWS_ACCESS_KEY_ID_ENV_KEY = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY_ENV_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
REGION_NAME = "us-east-1"

"""
Data Ingestion related constant start with DATA_INGESTION VAR NAME
"""
DATA_INGESTION_COLLECTION_NAME: str = "vehicleInsurance"
DATA_INGESTION_DATABASE_NAME: str = "vehicle"
DATA_INGESTION_DIR_NAME: str = "data_ingestion"
DATA_INGESTION_FEATURE_STORE_DIR: str = "feature_store"
DATA_INGESTION_INGESTED_DIR: str = "ingested"
DATA_INGESTION_TRAIN_TEST_SPLIT_RATIO: float = 0.25

"""
Data Validation realted constant start with DATA_VALIDATION VAR NAME
"""
DATA_VALIDATION_DIR_NAME: str = "data_validation"
DATA_VALIDATION_REPORT_FILE_NAME: str = "report.yaml"

"""
Data Transformation ralated constant start with DATA_TRANSFORMATION VAR NAME
"""
DATA_TRANSFORMATION_DIR_NAME: str = "data_transformation"
DATA_TRANSFORMATION_TRANSFORMED_DATA_DIR: str = "transformed"
DATA_TRANSFORMATION_TRANSFORMED_OBJECT_DIR: str = "transformed_object"

"""
MODEL TRAINER related constant start with MODEL_TRAINER var name
"""
MODEL_TRAINER_DIR_NAME: str = "model_trainer"
MODEL_TRAINER_TRAINED_MODEL_DIR: str = "trained_model"
MODEL_TRAINER_TRAINED_MODEL_NAME: str = "model.pkl"
MODEL_TRAINER_EXPECTED_SCORE: float = 0.7
MODEL_TRAINER_MODEL_CONFIG_FILE_PATH: str = os.path.join("schema", "model.yaml")
MODEL_TRAINER_N_ESTIMATORS: int = 150
MODEL_TRAINER_MIN_SAMPLES_SPLIT: int = 7
MODEL_TRAINER_MIN_SAMPLES_LEAF: int = 6
MODEL_TRAINER_MAX_DEPTH: int = 20
MODEL_TRAINER_CRITERION: str = 'entropy'
MODEL_TRAINER_MAX_FEATURES: str = 'sqrt'
MODEL_TRAINER_BOOTSTRAP: bool = True
MODEL_TRAINER_OOB_SCORE: bool = True
MODEL_TRAINER_RANDOM_STATE: int = 101


"""
MODEL Evaluation related constants
"""
MODEL_EVALUATION_CHANGED_THRESHOLD_SCORE: float = 0.02
MODEL_BUCKET_NAME = "insurance-claim-prediction"
MODEL_PUSHER_S3_KEY = "model-registry"


"""
APP related constants
"""
APP_HOST = "0.0.0.0"
APP_PORT = 5000