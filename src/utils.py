import os
import sys
# import numpy as np
# import pandas as pd
import dill
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score
from src.exception import CustomException


def save_object(file_path, obj):
    '''
    This function saves a given object to a given location.
    '''
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)
    
def evaluate_models(X_train, y_train, X_test, y_test, models: dict, params: dict) -> list:
    '''
    This function takes in data and a dictionary of models in order to evaluate each
    and returns a list of tuples with each model's name and score.
    It also uses a dict of hyperparameters to run a grid search on each model and uses the best performing ones.
    '''
    try:
        report = {}
        for model_name, model in models.items():
            
            # Tuning hyperparams using grid search
            param = params[model_name]
            gs = GridSearchCV(model, param, cv=3)
            gs.fit(X_train,y_train)

            # Fit data to model using the best performing parameters
            model.set_params(**gs.best_params_)
            model.fit(X_train, y_train)

            # predictions_train = model.predict(X_train)
            # model_score_train = r2_score(y_train, predictions_train)

            predictions_test = model.predict(X_test)
            model_score_test = r2_score(y_test, predictions_test)

            report[model_name] = model_score_test

        return sorted(report.items(), key=lambda kvp: kvp[1], reverse=True)

    except Exception as e:
        raise CustomException(e, sys)
    
def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return dill.load(file_obj)
    
    except Exception as e:
        raise CustomException(e, sys)