import os
import sys

import pickle
import dill
from sklearn.metrics import r2_score
from src.logger import logging

from sklearn.base import BaseEstimator, RegressorMixin, clone

from src.exception import CustomException

from sklearn.model_selection import GridSearchCV

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        print(f"Attempting to save object at: {file_path}")
        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)
        print(f"Object saved successfully at: {file_path}")
    except Exception as e:
        print(f"Error: {e}")

def evaluate_models(X_train, y_train, X_test, y_test, models, param):
    try:
        report = {}
        for model_name, model in models.items():
            logging.info(f"Evaluating model: {model_name}")
            
            try:
                para = param.get(model_name, {})
                model_to_use = clone(model)

                if para:
                    gs = GridSearchCV(model_to_use, para, cv=3)
                    gs.fit(X_train, y_train)
                    best_model = gs.best_estimator_
                else:
                    best_model = model_to_use
                    best_model.fit(X_train, y_train)

                y_train_pred = best_model.predict(X_train)
                y_test_pred = best_model.predict(X_test)

                train_score = r2_score(y_train, y_train_pred)
                test_score = r2_score(y_test, y_test_pred)

                logging.info(f"{model_name} - Train R2: {train_score}, Test R2: {test_score}")
                report[model_name] = test_score
            except Exception as e:
                logging.warning(f"Model {model_name} failed: {str(e)}")

        return report
    except Exception as e:
        logging.error(f"Error during model evaluation: {str(e)}")
        raise CustomException(e, sys)
    
def load_object(file_path):
    try:
        with open(file_path,"rb") as file_obj:
            return dill.load(file_obj)
            
    except Exception as e:
        raise CustomException(e,sys)
