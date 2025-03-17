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
MONGODB_URL_KEY = os.getenv("MONGO_DB_URL")

TARGET_COLUMN = "outcome"
PIPELINE_NAME: str = "src"
ARTIFACT_DIR: str = "artifacts"
FILE_NAME: str = "car_insurance.csv"

TRAIN_FILE_NAME: str = "train.csv"
TEST_FILE_NAME: str = "test.csv"

SCHEMA_FILE_PATH = os.path.join("schema", "schema.yaml")
# MODEL_FILE_NAME = "model.pkl"
# CURRENT_YEAR = date.today().year
# PREPROCSSING_OBJECT_FILE_NAME = "preprocessing.pkl"

"""
Data Ingestion related constant start with DATA_INGESTION VAR NAME
"""
DATA_INGESTION_COLLECTION_NAME: str = "vehicleInsurance"
DATA_INGESTION_DATABASE_NAME: str = "vehicle"
DATA_INGESTION_DIR_NAME: str = "data_ingestion"
DATA_INGESTION_FEATURE_STORE_DIR: str = "feature_store"
DATA_INGESTION_INGESTED_DIR: str = "ingested"
DATA_INGESTION_TRAIN_TEST_SPLIT_RATIO: float = 0.2

"""
Data Validation realted constant start with DATA_VALIDATION VAR NAME
"""
DATA_VALIDATION_DIR_NAME: str = "data_validation"
DATA_VALIDATION_REPORT_FILE_NAME: str = "report.yaml"
