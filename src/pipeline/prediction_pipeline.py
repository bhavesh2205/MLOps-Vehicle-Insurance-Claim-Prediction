import sys
from src.entity.config_entity import VehiclePredictorConfig
from src.entity.s3_estimator import ModelEstimator
from src.exception.exception import VehicleInsuranceException
from src.logging.logger import logging
from pandas import DataFrame
from typing import List


class VehicleData:
    def __init__(self,
        driving_experience,
        education,
        income,
        vehicle_year_before_2015,
        credit_score,
        annual_mileage,
        age,
        gender,
        vehicle_ownership,
        married,
        children,
        speeding_violations,
        past_accidents,
    ):
        """
        Vehicle Data constructor
        Input: all features of the trained model for prediction
        """
        try:
            self.driving_experience = driving_experience
            self.education = education
            self.income = income
            self.vehicle_year = vehicle_year_before_2015
            self.credit_score = credit_score
            self.annual_mileage = annual_mileage
            self.age = age
            self.gender = gender
            self.vehicle_ownership = vehicle_ownership
            self.married = married
            self.children = children
            self.speeding_violations = speeding_violations
            self.past_accidents = past_accidents

        except Exception as e:
            raise VehicleInsuranceException(e, sys) from e

    def get_vehicle_input_data_frame(self) -> DataFrame:
        """
        This function returns a DataFrame from VehicleData class input
        """
        try:
            vehicle_input_dict = self.get_vehicle_data_as_dict()
            return DataFrame(vehicle_input_dict)

        except Exception as e:
            raise VehicleInsuranceException(e, sys) from e

    def get_vehicle_data_as_dict(self):
        """
        This function returns a dictionary from VehicleData class input
        """
        logging.info("Entered get_vehicle_data_as_dict method as VehicleData class")

        try:
            input_data = {
                "driving_experience": [self.driving_experience],
                "education": [self.education],
                "income": [self.income],
                "vehicle_year": [self.vehicle_year],
                "credit_score": [self.credit_score],
                "annual_mileage": [self.annual_mileage],
                "age": [self.age],
                "gender": [self.gender],
                "vehicle_ownership": [self.vehicle_ownership],
                "married": [self.married],
                "children": [self.children],
                "speeding_violations": [self.speeding_violations],
                "past_accidents": [self.past_accidents],
            }

            logging.info("Created vehicle data dict")
            logging.info("Exited get_vehicle_data_as_dict method as VehicleData class")
            return input_data

        except Exception as e:
            raise VehicleInsuranceException(e, sys) from e


class VehicleDataClassifier:
    def __init__(self,prediction_pipeline_config: VehiclePredictorConfig = VehiclePredictorConfig(),) -> None:
        """
        :param prediction_pipeline_config: Configuration for prediction the value
        """
        try:
            self.prediction_pipeline_config = prediction_pipeline_config
        except Exception as e:
            raise VehicleInsuranceException(e, sys)

    def predict(self, dataframe) -> str:
        """
        This is the method of VehicleDataClassifier
        Returns: Prediction in string format
        """
        try:
            logging.info("Entered predict method of VehicleDataClassifier class")
            model = ModelEstimator(
                bucket_name=self.prediction_pipeline_config.model_bucket_name,
                model_path=self.prediction_pipeline_config.model_file_path,
            )
            result = model.predict(dataframe)

            return result
        except Exception as e:
            raise VehicleInsuranceException(e, sys)
