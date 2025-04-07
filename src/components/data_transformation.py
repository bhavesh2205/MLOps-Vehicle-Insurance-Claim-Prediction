import sys, os
import numpy as np
import pandas as pd

from imblearn.combine import SMOTEENN
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer

from src.constants.constant import TARGET_COLUMN, SCHEMA_FILE_PATH
from src.entity.config_entity import DataTransformationConfig
from src.entity.artifact_entity import (
    DataTransformationArtifact,
    DataIngestionArtifact,
    DataValidationArtifact,
)
from src.exception.exception import VehicleInsuranceException
from src.logging.logger import logging

from src.utils.main_utils import save_object, save_numpy_array_data, read_yaml_file


class DataTransformation:
    def __init__(
        self,
        data_ingestion_artifact: DataIngestionArtifact,
        data_transformation_config: DataTransformationConfig,
        data_validation_artifact: DataValidationArtifact,
    ):
        try:
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_transformation_config = data_transformation_config
            self.data_validation_artifact = data_validation_artifact
            self._schema_config = read_yaml_file(file_path=SCHEMA_FILE_PATH)
        except Exception as e:
            raise VehicleInsuranceException(e, sys)

    @staticmethod
    def read_data(file_path) -> pd.DataFrame:
        """Reads a CSV file and returns a DataFrame."""
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise VehicleInsuranceException(e, sys)

    def get_data_transformer_object(self) -> Pipeline:
        """
        Creates and returns a data transformation pipeline,
        including categorical encoding, missing value imputation, and standardization.
        """
        logging.info("Entered get_data_transformer_object method of DataTransformation class")

        try:
            # load schema configurations
            # num_features = self._schema_config['num_features']
            cat_features_ordinal = self._schema_config["ordinal_features"]
            cat_features_onehot = self._schema_config["onehot_features"]
            impute_features = self._schema_config["impute_features"]

            logging.info("Loaded column configurations from schema.")

            # define transformers
            imputer = SimpleImputer(strategy="mean")
            
            ordinal_encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
            
            onehot_encoder = OneHotEncoder(handle_unknown="ignore", drop="first")

            # creating ColumnTransformer pipeline
            preprocessor = ColumnTransformer(
                transformers=[
                    (
                        "ordinal",
                        ordinal_encoder,
                        cat_features_ordinal,
                    ),  # Ordinal encoding
                    (
                        "onehot",
                        onehot_encoder,
                        cat_features_onehot,
                    ),  # One-hot encoding
                    (
                        "impute",
                        imputer,
                        impute_features,
                    ),  # Mean imputation for missing values
                    # ("scaler", scaler, num_features)  # Standard scaling
                ],
                remainder="passthrough",
            )  # Keep other columns unchanged

            # wrapping everything in a single pipeline
            final_pipeline = Pipeline(steps=[("Preprocessor", preprocessor)])
            logging.info("Final Transformation Pipeline Ready!!")
            logging.info( "Exited get_data_transformer_object method of DataTransformation class")
            return final_pipeline

        except Exception as e:
            logging.exception("Exception occurred in get_data_transformer_object method of DataTransformation class")
            raise VehicleInsuranceException(e, sys) from e

    def drop_column(self, df):
        """Drop the specified columns if they exist."""
        logging.info("Dropping specified columns")
        drop_cols = self._schema_config["drop_columns"]
        # Check which columns exist in the DataFrame
        existing_cols = [col for col in drop_cols if col in df.columns]
        if existing_cols:
            df = df.drop(existing_cols, axis=1)
            logging.info(f"Dropped columns: {existing_cols}")
        return df

    def convert_credit_score(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Converts the Credit Score column from a 0-1 range to a 300-850 scale.
        """
        try:
            logging.info("Converting Credit Score to actual scale (300-850).")

            if "credit_score" in df.columns:
                df["credit_score"] = round(df["credit_score"] * (850 - 300) + 300, 2)

                logging.info("Credit Score conversion successful.")
                return df
        except Exception as e:
            logging.exception("Error in converting Credit Score.")
            raise VehicleInsuranceException(e, sys)

    def initiate_data_transformation(self) -> DataTransformationArtifact:
        """
        Initiates the data transformation component for the pipeline.
        """
        try:
            logging.info("Data Transformation Started !")
            if not self.data_validation_artifact.validation_status:
                raise Exception(self.data_validation_artifact.message)

            # load train and test data
            train_df = self.read_data(file_path=self.data_ingestion_artifact.trained_file_path)
            test_df = self.read_data(file_path=self.data_ingestion_artifact.test_file_path)
            logging.info("Train-Test data loaded")

            input_feature_train_df = train_df.drop(columns=[TARGET_COLUMN], axis=1)
            target_feature_train_df = train_df[TARGET_COLUMN]

            input_feature_test_df = test_df.drop(columns=[TARGET_COLUMN], axis=1)
            target_feature_test_df = test_df[TARGET_COLUMN]
            logging.info("Input and Target cols defined for both train and test df.")

            # apply custom transformations in specified sequence
            input_feature_train_df = self.drop_column(input_feature_train_df)
            input_feature_train_df = self.convert_credit_score(input_feature_train_df)

            input_feature_test_df = self.drop_column(input_feature_test_df)
            input_feature_test_df = self.convert_credit_score(input_feature_test_df)
            logging.info("Custom transformations applied to train and test data")

            logging.info("Starting data transformation")
            preprocessor = self.get_data_transformer_object()
            logging.info("Got the preprocessor object")

            logging.info("Initializing transformation for Training-data")
            input_feature_train_arr = preprocessor.fit_transform(input_feature_train_df)
            logging.info("Initializing transformation for Testing-data")
            input_feature_test_arr = preprocessor.transform(input_feature_test_df)
            logging.info("Transformation done end to end to train-test df.")

            logging.info("Applying SMOTEENN for handling imbalanced dataset.")
            smt = SMOTEENN(sampling_strategy="minority")
            input_feature_train_final, target_feature_train_final = smt.fit_resample(
                input_feature_train_arr, target_feature_train_df
            )
            input_feature_test_final, target_feature_test_final = smt.fit_resample(
                input_feature_test_arr, target_feature_test_df
            )
            logging.info("SMOTEENN applied to train-test df.")

            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]
            logging.info("feature-target concatenation done for train-test df.")

            save_object(
                self.data_transformation_config.transformed_object_file_path,
                preprocessor,
            )
            save_numpy_array_data(
                self.data_transformation_config.transformed_train_file_path,
                array=train_arr,
            )
            save_numpy_array_data(
                self.data_transformation_config.transformed_test_file_path,
                array=test_arr,
            )
            logging.info("Saving transformation object and transformed files.")

            # Get feature names from the preprocessor
            feature_names = preprocessor.get_feature_names_out()
            # convert to proper column names by replacing transformer names
            feature_names = [name.split("__")[-1] for name in feature_names]

            # convert the transformed NumPy arrays back to DataFrame
            train_transformed_df = pd.DataFrame(
                train_arr,
                columns=feature_names + [TARGET_COLUMN],
            )
            test_transformed_df = pd.DataFrame(
                test_arr,
                columns=feature_names + [TARGET_COLUMN],
            )

            # define paths for CSV files
            transformed_train_csv_path = os.path.join(os.path.dirname(self.data_transformation_config.transformed_train_file_path),
                "transformed_train.csv",
            )
            transformed_test_csv_path = os.path.join(os.path.dirname(self.data_transformation_config.transformed_test_file_path),
                "transformed_test.csv",
            )

            # save transformed data as CSV
            train_transformed_df.to_csv(
                transformed_train_csv_path, index=False, header=True
            )
            test_transformed_df.to_csv(
                transformed_test_csv_path, index=False, header=True
            )

            logging.info(f"Transformed train CSV saved at: {transformed_train_csv_path}")
            logging.info(f"Transformed test CSV saved at: {transformed_test_csv_path}")

            logging.info("Data transformation completed successfully")
            return DataTransformationArtifact(
                transformed_object_file_path=self.data_transformation_config.transformed_object_file_path,
                transformed_train_file_path=self.data_transformation_config.transformed_train_file_path,
                transformed_test_file_path=self.data_transformation_config.transformed_test_file_path,
            )
        except Exception as e:
            raise VehicleInsuranceException(e, sys) from e
