from src.entity.config_entity import ModelEvaluationConfig
from src.entity.artifact_entity import (
    ModelTrainerArtifact,
    DataTransformationArtifact,
    ModelEvaluationArtifact,
)
from sklearn.metrics import f1_score

from src.exception.exception import VehicleInsuranceException
from src.constants.constant import TARGET_COLUMN
from src.logging.logger import logging
from src.utils.main_utils import load_object

import sys
import pandas as pd
import numpy as np
from typing import Optional
from src.entity.s3_estimator import ModelEstimator
from dataclasses import dataclass


@dataclass
class EvaluateModelResponse:
    trained_model_f1_score: float
    best_model_f1_score: float
    is_model_accepted: bool
    difference: float


class ModelEvaluation:
    def __init__(
        self,
        model_eval_config: ModelEvaluationConfig,
        data_transformation_artifact: DataTransformationArtifact,
        model_trainer_artifact: ModelTrainerArtifact,
    ):
        try:
            self.model_eval_config = model_eval_config
            self.data_transformation_artifact = data_transformation_artifact
            self.model_trainer_artifact = model_trainer_artifact
        except Exception as e:
            raise VehicleInsuranceException(e, sys) from e

    def get_best_model(self) -> Optional[ModelEstimator]:
        """
        Method Name :   get_best_model
        Description :   This function is used to get model from production stage.

        Output      :   Returns model object if available in s3 storage
        On Failure  :   Write an exception log and then raise an exception
        """
        try:
            bucket_name = self.model_eval_config.bucket_name
            model_path = self.model_eval_config.s3_model_key_path
            model_estimator = ModelEstimator(
                bucket_name=bucket_name, model_path=model_path
            )

            if model_estimator.is_model_present():
                return model_estimator
            return None
        except Exception as e:
            raise VehicleInsuranceException(e, sys)

    def evaluate_model(self) -> EvaluateModelResponse:
        """
        Method Name :   evaluate_model
        Description :   This function is used to evaluate trained model
                        with production model and choose best model

        Output      :   Returns bool value based on validation results
        On Failure  :   Write an exception log and then raise an exception
        """
        try:
            # Load transformed test data
            test_arr = np.load(
                self.data_transformation_artifact.transformed_test_file_path
            )
            x = test_arr[:, :-1]  # All columns except last
            y = test_arr[:, -1]  # Last column is target

            logging.info("Transformed test data loaded and ready for prediction...")

            # Load trained model
            trained_model = load_object(
                file_path=self.model_trainer_artifact.trained_model_file_path
            )
            logging.info("Trained model loaded/exists.")

            # Get predictions from new trained model directly from numpy array
            # This is safe because MyModel.predict uses preprocessing_object.transform
            # which can handle numpy arrays directly
            y_hat_trained_model = trained_model.trained_model_object.predict(x)
            trained_model_f1_score = f1_score(y, y_hat_trained_model)
            logging.info(f"F1_Score for this model: {trained_model_f1_score}")

            best_model_f1_score = None
            best_model = self.get_best_model()
            if best_model is not None:
                logging.info(f"Computing F1_Score for production model..")
                # Use the trained_model_object directly to bypass the preprocessing steps
                # that are causing column name issues
                if best_model.loaded_model is None:
                    best_model.loaded_model = best_model.load_model()
                y_hat_best_model = best_model.loaded_model.trained_model_object.predict(
                    x
                )
                best_model_f1_score = f1_score(y, y_hat_best_model)
                logging.info(
                    f"F1_Score-Production Model: {best_model_f1_score}, F1_Score-New Trained Model: {trained_model_f1_score}"
                )

            tmp_best_model_score = (
                0 if best_model_f1_score is None else best_model_f1_score
            )
            result = EvaluateModelResponse(
                trained_model_f1_score=trained_model_f1_score,
                best_model_f1_score=best_model_f1_score,
                is_model_accepted=trained_model_f1_score > tmp_best_model_score,
                difference=trained_model_f1_score - tmp_best_model_score,
            )
            logging.info(f"Result: {result}")
            return result

        except Exception as e:
            raise VehicleInsuranceException(e, sys)

    def initiate_model_evaluation(self) -> ModelEvaluationArtifact:
        """
        Method Name :   initiate_model_evaluation
        Description :   This function is used to initiate all steps of the model evaluation

        Output      :   Returns model evaluation artifact
        On Failure  :   Write an exception log and then raise an exception
        """
        try:
            print(
                "------------------------------------------------------------------------------------------------"
            )
            logging.info("Initialized Model Evaluation Component.")
            evaluate_model_response = self.evaluate_model()
            s3_model_path = self.model_eval_config.s3_model_key_path

            model_evaluation_artifact = ModelEvaluationArtifact(
                is_model_accepted=evaluate_model_response.is_model_accepted,
                s3_model_path=s3_model_path,
                trained_model_path=self.model_trainer_artifact.trained_model_file_path,
                changed_accuracy=evaluate_model_response.difference,
            )

            logging.info(f"Model evaluation artifact: {model_evaluation_artifact}")
            return model_evaluation_artifact
        except Exception as e:
            raise VehicleInsuranceException(e, sys) from e
