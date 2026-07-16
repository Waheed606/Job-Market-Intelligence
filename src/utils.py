 
from sklearn.model_selection import RandomizedSearchCV, KFold
import os
import sys
import pandas as pd
import dill
from sklearn.metrics import accuracy_score

from src.exception import CustomeExpection


def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, 'wb') as file_obj:
            dill.dump(obj, file_obj)

    except Exception as e:
        raise CustomeExpection(e, sys)


def load_object(file_path):
    try:
        with open(file_path, 'rb') as file_obj:
            return dill.load(file_obj)

    except Exception as e:
        raise CustomeExpection(e, sys)


def evalaute_models(X_train, X_test, y_train, y_test, models, param, n_iter=8):
    try:
        report = {}
        model_names = list(models.keys())

        # Plain KFold instead of the classifier default (StratifiedKFold).
        # StratifiedKFold requires every class to have at least n_splits
        # members; this dataset has many classes with only 1-2 examples
        # (high-cardinality Title target), so KFold avoids that requirement.
        cv_strategy = KFold(n_splits=3, shuffle=True, random_state=42)

        for i in range(len(model_names)):
            model_name = model_names[i]
            model = models[model_name]
            param_grid = param[model_name]

            print(f"[{i+1}/{len(model_names)}] Tuning {model_name} ...")

            rs = RandomizedSearchCV(
                model,
                param_distributions=param_grid,
                n_iter=n_iter,
                cv=cv_strategy,
                scoring='accuracy',
                n_jobs=-1,
                verbose=1,
                random_state=42
            )
            rs.fit(X_train, y_train)

            model.set_params(**rs.best_params_)
            model.fit(X_train, y_train)

            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            train_model_score = accuracy_score(y_train, y_train_pred)
            test_model_score = accuracy_score(y_test, y_test_pred)

            print(f"    -> best params: {rs.best_params_} | test accuracy: {test_model_score:.4f}")

            report[model_name] = test_model_score

        return report

    except Exception as e:
        raise CustomeExpection(e, sys)