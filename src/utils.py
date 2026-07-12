from sklearn.model_selection import RandomizedSearchCV
import os
import sys 
import pandas as pd 
import dill
from src.exception import CustomeExpection
from sklearn.metrics import accuracy_score

from src.exception import CustomeExpection

def save_object(file_path,obj):
    try:

        dir_path=os.path.dirname(file_path) 
        os.makedirs(file_path,exist_ok=True)

        with open(file_path,'wb') as file_obj:
            dill.dum(obj,file_obj)

    except Exception as e:
        raise CustomeExpection(e)
    
def evalaute_models(X_train,X_test,y_train,y_test,models,param):
    try:
        report={}
        for i in range(len(list(models))):

            model=list(models.value())[i]
            param=list(models.keys())[i]
        
            rs = RandomizedSearchCV(
            model,
            param_distributions=param,
            # n_iter=20,
            cv=5,
            scoring='accuracy',
            # random_state=42,
            n_jobs=-1
        )   
            rs.fit(X_train,y_train)
            
            model.set_params(**rs.best_params_)

            model.fit(X_train,y_train)
            y_train_pred=model.predict(y_train)
            y_test_pred=model.predict(y_test)

            train_model_score=accuracy_score(y_train,y_train_pred)
            test_model_score=accuracy_score(y_test,y_test_pred)

            report[list(model.keys()[i])]=test_model_score

        return report
            

    except Exception as e :
        raise CustomeExpection(e,sys)