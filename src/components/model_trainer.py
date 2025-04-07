import mlflow
import sys, os

from src.exception.exception import VehicleInsuranceException
from src.logging.logger import logging
import numpy as np

from src.entity.config_entity import ModelTrainingConfig
from src.entity.artifact_entity import DataTransformationArtifact, ModelTrainerArtifact, ClassificationMetricArtifact
from src.utils.main_utils import load_numpy_array_data, load_object, save_object, evaluate_model
from src.utils.ml_utils.metric.classification_metric import get_classification_score
from src.utils.ml_utils.model.estimator import InsuranceModel

from sklearn.metrics import recall_score
from sklearn.ensemble import (
    AdaBoostClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier
)

import dagshub
dagshub.init(repo_owner='imbhavesh7', repo_name='MLOps-Vehicle-Insurance-Claim-Prediction', mlflow=True)

class ModelTrainer:
    def __init__(self, model_trainer_config: ModelTrainingConfig, data_transformation_artifact: DataTransformationArtifact):
        try:
            logging.info("Initializing ModelTrainer class")
            self.model_trainer_config = model_trainer_config
            self.data_transformation_artifact = data_transformation_artifact
        except Exception as e:
            logging.error(f"Error in initializing ModelTrainer: {str(e)}")
            raise VehicleInsuranceException(e, sys)
        
    def track_mlflow(self, best_model, classificationmetric):
            with mlflow.start_run():
                f1_score=classificationmetric.f1_score
                precision_score=classificationmetric.precision_score
                recall_score=classificationmetric.recall_score
                
                mlflow.log_metric("f1_score", f1_score)
                mlflow.log_metric("precision", precision_score)
                mlflow.log_metric("recall", recall_score)
                mlflow.sklearn.log_model(best_model,"model")
        
    def train_model(self, x_train, y_train, x_test, y_test):
        try:
            logging.info("Starting model training")
            models = {
                "Random Forest": RandomForestClassifier(verbose=1),
                "Gradient Boosting": GradientBoostingClassifier(verbose=1),
                "AdaBoost": AdaBoostClassifier()
            }
            params = {
                "Random Forest": {
                    'criterion': ['gini', 'entropy'],
                    'max_features': ['sqrt', 'log2'],
                    'n_estimators': [100, 200, 300],  
                    'max_depth': [None, 20, 30],  
                    'min_samples_split': [2, 5],  
                    'min_samples_leaf': [1, 2],  
                    'class_weight': ['balanced', 'balanced_subsample']  
                },
                "Gradient Boosting": {
                    'loss': ['log_loss'],
                    'learning_rate': [.05, .01],
                    'subsample': [0.7, 0.8, 0.9], 
                    'criterion': ['friedman_mse'],
                    'max_features': ['sqrt', 'log2'],
                    'n_estimators': [100, 200, 300],  
                    'max_depth': [4, 6, 8],  
                    'min_samples_split': [2, 5],  
                    'min_samples_leaf': [1, 2]
                },
                "AdaBoost": {
                    'learning_rate': [.05, .01],
                    'n_estimators': [100, 200, 300]
                }
            }

            logging.info("Evaluating models")
            model_report: dict = evaluate_model(x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test,
                                                models=models, param=params)
            
            best_model_score = max(sorted(model_report.values()))
            best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)]
            best_model = models[best_model_name]
            logging.info(f"Best model selected: {best_model_name} with score {best_model_score}")
            
            y_train_pred = best_model.predict(x_train) 
            classification_train_metric = get_classification_score(y_true=y_train, y_pred=y_train_pred)
            
            y_test_pred = best_model.predict(x_test)
            classification_test_metric = get_classification_score(y_true=y_test, y_pred=y_test_pred)
            
            ## track the experiment with mlflow
            self.track_mlflow(best_model,classification_train_metric)
            self.track_mlflow(best_model,classification_test_metric)

            logging.info("Loading preprocessor")
            preprocessor = load_object(file_path=self.data_transformation_artifact.transformed_object_file_path)
            
            model_dir_path = os.path.dirname(self.model_trainer_config.trained_model_file_path)
            os.makedirs(model_dir_path, exist_ok=True)
            
            logging.info("Saving trained model")
            insurance_model = InsuranceModel(preprocessor=preprocessor, model=best_model)
            save_object(self.model_trainer_config.trained_model_file_path, obj=insurance_model)
            
            save_object("models/model.pkl", best_model)
            
            model_trainer_artifact = ModelTrainerArtifact(
                trained_model_file_path=self.model_trainer_config.trained_model_file_path,
                train_metric_artifact=classification_train_metric,
                test_metric_artifact=classification_test_metric
            )
            
            logging.info("Model training completed successfully")
            return model_trainer_artifact
        except Exception as e:
            logging.error(f"Error in training model: {str(e)}")
            raise VehicleInsuranceException(e, sys)
    
    def initiate_model_trainer(self) -> ModelTrainerArtifact:
        try:
            logging.info("Initiating model training process")
            train_file_path = self.data_transformation_artifact.transformed_train_file_path 
            test_file_path = self.data_transformation_artifact.transformed_test_file_path
            
            logging.info("Loading training and testing arrays")
            train_arr = load_numpy_array_data(train_file_path)
            test_arr = load_numpy_array_data(test_file_path)
            
            x_train, y_train, x_test, y_test = (
                train_arr[:, :-1],
                train_arr[:, -1],
                test_arr[:, :-1],
                test_arr[:, -1],
            )
            
            logging.info("Starting model training")
            model_trainer_artifact = self.train_model(x_train, y_train, x_test, y_test)
            logging.info("Model training process completed successfully")
            return model_trainer_artifact
        except Exception as e:
            logging.error(f"Error in initiating model trainer: {str(e)}")
            raise VehicleInsuranceException(e, sys)
