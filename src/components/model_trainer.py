import sys
import os
from dataclasses import dataclass

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor

from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from sklearn.ensemble import AdaBoostRegressor, GradientBoostingRegressor, RandomForestRegressor

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_model

@dataclass
class ModelTrainerConfig:
    trained_model_path = os.path.join('artifacts', 'model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Splitting data into input and output")
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1], 
                train_array[:, -1], 
                test_array[:, :-1], 
                test_array[:, -1]
            )

            models = {
                "Random_Forest": RandomForestRegressor(),
                "Decision_Tree": DecisionTreeRegressor(),
                "Gradient_Boosting": GradientBoostingRegressor(),
                "Linear_Regression": LinearRegression(),
                "K_Neighbours": KNeighborsRegressor(),
                "Xgboost" : XGBRegressor(),
                "Catboost" : CatBoostRegressor(verbose=False),
                "Adaboost": AdaBoostRegressor()
            }

            model_report: dict = evaluate_model(X_train, y_train, X_test, y_test, models)

            best_model_name= max(model_report, key=model_report.get)
            best_model_score = model_report[best_model_name]
            best_model = models[best_model_name]

            if best_model_score < 0.6:
                raise CustomException("No best model found")
            logging.info("Best found model on train and test dataset")

            save_object(
                file_path = self.model_trainer_config.trained_model_path,
                obj=best_model
            )

            prediction = best_model.predict(X_test)
            r2score = r2_score(y_test, prediction)

            return r2score
        
        except Exception as e:
            raise CustomException(e, sys)
