from src.utils import save_object
from sklearn import preprocessing
from sklearn.impute import SimpleImputer
from pandas.core.arrays import categorical
import sys
import os
import pandas as pd 
import numpy as np 
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder,MultiLabelBinarizer
import re
import string

from  dataclasses import dataclass
from src.logger import logging
from src.exception import CustomeExpection

@dataclass
class DataTransfromationConfig:
    preprocess_pickle_file=os.path.join('artifacts','preprocssor,pkle')

class DataTransformationMethod:
    def __init__(self):
        pass
    try:
        logging.info('The data cleaning has been start')

        def text_cleaning(self,text):
            logging.info('text cleaning has been start')
            text=str(text)
            text=text.lower()
            text=re.sub(r'\s','',text)
            # text=text.strip()
            return text
        

        # def multilabelbinzar(self,value,df):
        #     logging.info('keyword multilabelbinarizer has start')
            
        #     mlb_keyword=MultiLabelBinarizer()
        #     keywords_encoded=mlb_keyword.fit_transform(df['value'])

        #     keywords_df=pd.DataFrame(
        #         keywords_encoded,
        #         columns=mlb_keyword.classes_,
        #         index=df.index
        #     )
        #     return keywords_df
        

        logging.info('data Encoding has been start')

        def label_encoding(self,value,df):

            logging.info(f'{value} encoding has been start')
            lb=LabelEncoder()
            df['value']=lb.fit_transform(df['value'])
            return df['value']
        
        # def onehot_encoding(self,value,df):

        #     logging.info(f'{value} encoding has been start')
        #     ohe=OneHotEncoder(sparse_output=False)
        #     encoded=ohe.fit_transform(df[['ExperienceLevel']])
        #     encoded_df = pd.DataFrame(
        #             encoded,
        #             columns=ohe.get_feature_names_out(['ExperienceLevel']),
        #             index=df.index
        #     )
        #     return encoded_df 
    except Exception as e:
        raise CustomeExpection(e,sys)

class DataTransformation():
    def __init__(self):
        self.data_transformation_config=DataTransfromationConfig()
        self.data_transformation_method=DataTransformationMethod()

    def get_data_transformation_obj(self):
        try:
            logging.info('the data taining pipelines is start')
            cat_columns=['Keywords','ExperinceLevel','Title']
            cat_pipelines=Pipeline(
                steps=[
                    # ('text_cleaning',self.data_transformation_method.text_cleaning()),
                    ('OnehotEncoding',OneHotEncoder(handle_unknown='ignore')),
                    # ('LabelEncoding',LabelEncoder()),
                    # ('multilabelbinarizer',MultiLabelBinarizer())
                ]
            )
            preprocesser=ColumnTransformer(
                transformers=[
                    'cat_pipelines',cat_pipelines,cat_columns
                ]
            )

            return preprocesser
        

        except Exception as e:
            raise CustomeExpection(e,sys)

    def InitiateDataTransformation(self,train_path,test_path):
        try:
            # train_df=pd.read_csv('/artifacts/train_csv') 
            # test_df=pd.read_csv('/artifacts/test_csv')
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)

            logging.info('Read training and testing completed')
            logging.info('obtaining propressing object')
            
            preprocessing_obj=self.get_data_transformation_obj()

            target_column='Title'

            # the responsibilites features is not use for the training model
            responsiblilties_columns='Responsiblilities'

            input_features_df_train=train_df.drop(columns=[target_column,responsiblilties_columns])
            target_features_df_train=train_df[target_column]

            input_features_df_test=test_df.drop(columns=[target_column,responsiblilties_columns])
            target_features_df_test=test_df[target_column]

            logging.info('Applying preprocessing object on  train and test dataframe')

            input_features_train_arr=preprocessing_obj.fit_transform(input_features_df_train)
            input_features_test_arr=preprocessing_obj.transform(input_features_df_test)

            train_arr=np.c_[
                input_features_train_arr,np.arrays(target_features_df_train)
            ]
            test_arr=[
                input_features_test_arr,np.arrays(target_features_df_test)
            ]

            save_object(file_path=self.data_transformation_config,obj=preprocessing_obj)

            return(
                train_arr,
                test_arr,
                self.data_transformation_config.preprocess_pickle_file
            )
            

        except Exception as e:
            raise CustomeExpection(e,sys)
