from notebook import params
from src.utils import save_object
from src.exception import CustomeExpection
from src.logger import logging
from src.utils import evalaute_models
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier

import os 
import sys 

from dataclasses import dataclass

@dataclass
class ModelTrainingConfig:
    model_train_file_path=os.path.join('artifacts','model.pkle')


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainingConfig()
    
    def initiate_model_trainer(train_arr,test_arr,preprocessor_path):
        try:
            logging.inf('Split the training and testing set')
            X_train,X_test,y_train,y_test=(
                train_arr[:,1],
                train_arr[:,1],
                test_arr[:,1],
                test_arr[:,1]
                )
            models={
                'LogisticRegression':LogisticRegression(),
                'DecisionTreeClassifier':DecisionTreeClassifier(),
                'RandomForestClassifier':RandomForestClassifier(),
                'MultinomialNB':MultinomialNB(),
                'KNeighborsClassifier':KNeighborsClassifier(),
                'GradientBoostingClassifier':GradientBoostingClassifier()
            }

            params={
                'LogisticRegression':{
                    'C': [0.01, 0.1, 1, 10, 100],
                    'solver': ['liblinear', 'lbfgs']
                },
                'RandomForestClassifier':{
                    'n_estimators': [100, 200, 300, 500],
                    'max_depth': [5, 10, 20, None],
                    'min_samples_split': [2, 5, 10]
                }
            }
            model_report:dict=evalaute_models(X_train=X_train,y_train=y_train,models=models,param=params)

            # to get best model score from the dictionary
            best_model_score=max(sorted(model_report.values()))
            
            # to get best model name from the dictionary
            best_model_name=list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]

            best_model=models[best_model_name]

            if best_model_score<0.7:
                raise CustomeExpection('No best model found')
            logging.info('No best models found in both the test and train dataset')

            save_object(
                file_path=self.model_trainer_config.model_train_file_path,obj=best_model
            )

            predicted=best_model.predict(X_test)

        except Exception as e:
            raise CustomeExpection(e,sys)

        