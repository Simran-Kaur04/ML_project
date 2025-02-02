import sys
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessed_obj_file_path = os.path.join("artifacts", "preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_obj(self):
        try:
            num_features = ["reading_score", "writing_score"]
            cat_features = ["gender","race_ethnicity","parental_level_of_education","lunch","test_preparation_course"]
            num_pipeline = Pipeline(
                [('imputer', SimpleImputer(strategy='median')),
                    ('scaler', StandardScaler(with_mean=False))]
            )

            logging.info("Pipeline for numerical features created")

            cat_pipeline = Pipeline(
                [('imputer', SimpleImputer(strategy='most_frequent')),
                    ('encoding', OneHotEncoder()),
                    ('scaler', StandardScaler(with_mean=False))]
            )

            logging.info("Pipeline for categorical features created")

            preprocessor = ColumnTransformer(
                [
                    ('num_pipeline', num_pipeline, num_features),
                    ('cat_pipeline', cat_pipeline, cat_features)
                ]
            )

            return preprocessor
        
        except Exception as e:
            raise CustomException(e, sys)
        
    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Reading train and test data completed")

            preprocessing_obj = self.get_data_transformer_obj()
            target_col = 'math_score'

            input_features_train_df = train_df.drop(columns = [target_col], axis=1)
            target_feature_train_df = train_df[target_col]

            input_features_test_df = test_df.drop(columns=[target_col], axis=1)
            target_feature_test_df = test_df[target_col]

            logging.info("Applying preprocessing on train and test dataframe")

            input_features_train_arr = preprocessing_obj.fit_transform(input_features_train_df)
            input_features_test_arr = preprocessing_obj.transform(input_features_test_df)

            train_arr = np.c_[input_features_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_features_test_arr, np.array(target_feature_test_df)]

            logging.info("Saved preprocessing object")

            save_object(
                file_path=self.data_transformation_config.preprocessed_obj_file_path,
                obj=preprocessing_obj
            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessed_obj_file_path
            )
        except Exception as e:
            raise CustomException(e, sys)
    