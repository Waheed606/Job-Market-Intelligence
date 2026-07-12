import sys
from src.exception import CustomeExpection
from src.logger import logging 
from src.utils import load_object
import pandas as pd 

class Pred_Pipelines:
    try:
        def __init__(self,features):
            model_path=('artifacts/model.pkl')
            preprocessor_path=('artifacts/preprocssor.pkl')
            model=load_object(file_path=model_path)
            preprocessor=load_object(file_path=preprocessor_path)
            data_scaled=preprocessor.transform(features)
            predition=model.predict(data_scaled)
            return predition
            
    except Exception as e:
        raise CustomeExpection(e,sys)


        

class CustomData:
    logging.info('User input are convert in the dataframe')
    def __init__(self,keywords:str,experincelevel:str):
        self.keywords=keywords,
        self.experiencelevel=experincelevel

    def  get_data_frame(self):
        try:
            custom_data_frame={
                'keywords':[self.keywods],
                'experiencelevel':[self.experiencelevel]
            }
            return pd.DataFrame(custom_data_frame)
        except Exception as e:
            raise CustomeExpection(e,sys)


    
