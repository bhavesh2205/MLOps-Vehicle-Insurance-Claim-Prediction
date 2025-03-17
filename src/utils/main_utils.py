import yaml
import os, sys
from src.exception.exception import VehicleInsuranceException
from src.logging.logger import logging
from src.constants.constant import SCHEMA_FILE_PATH
import numpy as np
import dill
import pickle


def read_yaml_file(file_path: str) -> dict:
    """
    Method Name :   read_yaml_file
    Description :   This method reads the yaml file and returns the configuration as dictionary
    Output      :   Returns the configuration as dictionary
    On Failure  :   Write an exception log and then raise an exception
    """
    try:
        with open(file_path, "rb") as yaml_file:
            return yaml.safe_load(yaml_file)
    except Exception as e:
        raise VehicleInsuranceException(e, os)
