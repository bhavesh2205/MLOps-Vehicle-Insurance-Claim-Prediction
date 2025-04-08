from src.cloud_storage.aws_storage import SimpleStorageService
from src.exception.exception import VehicleInsuranceException
from src.logging.logger import logging
from src.entity.estimator import MyModel
import sys
from pandas import DataFrame


class ModelEstimator:
    """
    This class is used to save and retrieve our model from s3 bucket and to do prediction
    """

    def __init__(
        self,
        bucket_name,
        model_path,
    ):
        """
        :param bucket_name: Name of your model bucket
        :param model_path: Location of your model in bucket
        """
        self.bucket_name = bucket_name
        self.s3 = SimpleStorageService()
        self.model_path = model_path
        self.loaded_model: MyModel = None

    def is_model_present(self):
        """
        Check if model exists at the specified path in S3 bucket
        :return: True if model exists, False otherwise
        """
        try:
            return self.s3.s3_key_path_available(
                bucket_name=self.bucket_name, s3_key=self.model_path
            )
        except VehicleInsuranceException as e:
            logging.error("Error checking if model is present", exc_info=True)
            return False

    def load_model(self) -> MyModel:
        """
        Load the model from the model_path
        :return:
        """

        return self.s3.load_model(self.model_path, bucket_name=self.bucket_name)

    def save_model(self, from_file, remove: bool = False) -> None:
        """
        Save the model to the model_path
        :param from_file: Your local system model path
        :param remove: By default it is false that mean you will have your model locally available in your system folder
        :return:
        """
        try:
            self.s3.upload_file(
                from_file,
                to_filename=self.model_path,
                bucket_name=self.bucket_name,
                remove=remove,
            )
        except Exception as e:
            raise VehicleInsuranceException(e, sys)

    def predict(self, dataframe: DataFrame):
        """
        Make predictions using the model loaded from S3

        :param dataframe: DataFrame containing the features for prediction
        :return: NumPy array with the model's predictions
        :raises VehicleInsuranceException: If an error occurs during prediction
        """
        try:
            if self.loaded_model is None:
                self.loaded_model = self.load_model()
            return self.loaded_model.predict(dataframe=dataframe)
        except Exception as e:
            raise VehicleInsuranceException(e, sys)
