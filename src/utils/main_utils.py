import yaml
import os, sys
import dill

from imblearn.over_sampling import SMOTE
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import recall_score

from src.exception.exception import VehicleInsuranceException
from src.logging.logger import logging
#from src.constants.constant import SCHEMA_FILE_PATH
import numpy as np
#mport pickle



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


def save_object(file_path: str, obj: object) -> None:
    logging.info("Entered the save_object method of utils")

    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "wb") as file_object:
            dill.dump(obj, file_object)

        logging.info("Exited the save_object method of utils")
    except Exception as e:
        raise VehicleInsuranceException(e, sys) from e


def save_numpy_array_data(file_path: str, array: np.array):
    """
    Save numpy array data to file
    file_path: str location of file to save
    array: np.array data to save
    """
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, "wb") as file_object:
            np.save(file_object, array)
    except Exception as e:
        raise VehicleInsuranceException(e, sys) from e


def applysmote(X, y):
        """
        Applies SMOTE to balance the dataset by oversampling the minority class.
        """
        try:
            logging.info("Applying SMOTE to handle class imbalance.")
            smote = SMOTE(random_state=42)
            X_resampled, y_resampled = smote.fit_resample(X, y)
            logging.info("SMOTE applied successfully.")
            return X_resampled, y_resampled
        except Exception as e:
            logging.exception("Error in applying SMOTE.")
            raise VehicleInsuranceException(e, sys)
        
def load_object(file_path: str) -> object:
    """
    Returns model/object from project directory.
    file_path: str location of file to load
    return: Model/Obj
    """
    try:
        with open(file_path, "rb") as file_obj:
            obj = dill.load(file_obj)
        return obj
    except Exception as e:
        raise VehicleInsuranceException(e, sys)

def save_numpy_array_data(file_path: str, array: np.array):
    """
    Save numpy array data to file
    file_path: str location of file to save
    array: np.array data to save
    """
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, 'wb') as file_obj:
            np.save(file_obj, array)
    except Exception as e:
        raise VehicleInsuranceException(e, sys) from e


def load_numpy_array_data(file_path: str) -> np.array:
    """
    load numpy array data from file
    file_path: str location of file to load
    return: np.array data loaded
    """
    try:
        with open(file_path, 'rb') as file_obj:
            return np.load(file_obj)
    except Exception as e:
        raise VehicleInsuranceException(e, sys) from e


def evaluate_model(x_train,y_train,x_test, y_test, models, param):
    try:
        report = {}
        
        for i in range(len(list(models))):
            model=list(models.values())[i]
            para=param[list(models.keys())[i]]
            
            gs_cv = GridSearchCV(model,para,cv=3)
            gs_cv.fit(x_train,y_train)
            
            model.set_params(**gs_cv.best_params_)
            model.fit(x_train,y_train)
            
            y_train_pred=model.predict(x_train)
            
            y_test_pred=model.predict(x_test)
            
            train_model_score=recall_score(y_train, y_train_pred)
            test_model_score=recall_score(y_test, y_test_pred)
            
            report[list(models.keys())[i]] = test_model_score
            
            return report
    except Exception as e:
        raise VehicleInsuranceException(e,sys)   
