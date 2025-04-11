import unittest
import sys
import os
from src.pipeline.training_pipeline import TrainPipeline
from src.exception.exception import VehicleInsuranceException


class TestTrainingPipeline(unittest.TestCase):
    def setUp(self):
        """
        Set up test fixtures before each test method.
        """
        self.pipeline = TrainPipeline()

    def test_pipeline_initialization(self):
        """
        Test if pipeline initializes correctly with all required configurations.
        """
        self.assertIsNotNone(self.pipeline.data_ingestion_config)
        self.assertIsNotNone(self.pipeline.data_validation_config)
        self.assertIsNotNone(self.pipeline.data_transformation_config)
        self.assertIsNotNone(self.pipeline.model_trainer_config)
        self.assertIsNotNone(self.pipeline.model_evaluation_config)
        self.assertIsNotNone(self.pipeline.model_pusher_config)

    def test_data_ingestion(self):
        """
        Test data ingestion component.
        """
        try:
            artifact = self.pipeline.start_data_ingestion()
            self.assertIsNotNone(artifact)
            self.assertTrue(os.path.exists(artifact.trained_file_path))
            self.assertTrue(os.path.exists(artifact.test_file_path))
        except VehicleInsuranceException as e:
            self.fail(f"Data ingestion failed: {str(e)}")

    def test_data_validation(self):
        """
        Test data validation component.
        """
        try:
            ingestion_artifact = self.pipeline.start_data_ingestion()
            validation_artifact = self.pipeline.start_data_validation(
                ingestion_artifact
            )
            self.assertIsNotNone(validation_artifact)
            self.assertTrue(hasattr(validation_artifact, "validation_status"))
        except VehicleInsuranceException as e:
            self.fail(f"Data validation failed: {str(e)}")

    def test_pipeline_integration(self):
        """
        Test the complete pipeline integration.
        """
        try:
            self.pipeline.run_pipeline()
            # If we reach here, the pipeline completed without exceptions
            self.assertTrue(True)
        except VehicleInsuranceException as e:
            self.fail(f"Pipeline integration test failed: {str(e)}")


if __name__ == "__main__":
    unittest.main()
