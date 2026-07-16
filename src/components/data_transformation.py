from src.utils import save_object
import sys
import os
import re
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, MultiLabelBinarizer
from sklearn.base import BaseEstimator, TransformerMixin

from dataclasses import dataclass
from src.logger import logging
from src.exception import CustomeExpection



class KeywordsMultiLabelBinarizer(BaseEstimator, TransformerMixin):

    def __init__(self):
        self.mlb = MultiLabelBinarizer()

    def _prepare(self, X):
        # ColumnTransformer passes a 1-column DataFrame when given ['Keywords']
        series = X.iloc[:, 0] if isinstance(X, pd.DataFrame) else pd.Series(X)
        return series.str.split(';')

    def fit(self, X, y=None):
        self.mlb.fit(self._prepare(X))
        return self

    def transform(self, X):
        return self.mlb.transform(self._prepare(X))

    def get_feature_names_out(self, input_features=None):
        return self.mlb.classes_


@dataclass
class DataTransfromationConfig:
    preprocess_pickle_file = os.path.join('artifacts', 'preprocessor.pkl')
    target_encoder_pickle_file = os.path.join('artifacts', 'target_encoder.pkl')


class DataTransformation():
    def __init__(self):
        self.data_transformation_config = DataTransfromationConfig()

    def get_data_transformation_obj(self):
        try:
            logging.info('building the data transformation pipeline')

            preprocesser = ColumnTransformer(
                transformers=[
                    ('keywords_mlb', KeywordsMultiLabelBinarizer(), ['Keywords']),
                    ('experience_ohe', OneHotEncoder(handle_unknown='ignore', sparse_output=False), ['ExperienceLevel']),
                ]
            )
            return preprocesser

        except Exception as e:
            raise CustomeExpection(e, sys)

    def InitiateDataTransformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            logging.info('Read training and testing completed')

            target_column = 'Title'
            job_id_column = 'JobID'
            responsiblilties_columns = 'Responsibilities'

            # clean the target exactly like the notebook (lowercase + strip whitespace)
            train_df[target_column] = train_df[target_column]
            test_df[target_column] = test_df[target_column]

            drop_columns = [target_column, responsiblilties_columns, job_id_column]

            input_features_df_train = train_df.drop(columns=drop_columns)
            input_features_df_test = test_df.drop(columns=drop_columns)

            target_encoder = LabelEncoder()
            all_titles = pd.concat([train_df[target_column], test_df[target_column]], axis=0)
            target_encoder.fit(all_titles)

            target_features_train = target_encoder.transform(train_df[target_column])
            target_features_test = target_encoder.transform(test_df[target_column])

            preprocessing_obj = self.get_data_transformation_obj()

            logging.info('applying preprocessing object on train and test dataframe')
            input_features_train_arr = preprocessing_obj.fit_transform(input_features_df_train)
            input_features_test_arr = preprocessing_obj.transform(input_features_df_test)

            train_arr = np.c_[input_features_train_arr, target_features_train]
            test_arr = np.c_[input_features_test_arr, target_features_test]

            save_object(file_path=self.data_transformation_config.preprocess_pickle_file, obj=preprocessing_obj)
            save_object(file_path=self.data_transformation_config.target_encoder_pickle_file, obj=target_encoder)

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocess_pickle_file
            )

        except Exception as e:
            raise CustomeExpection(e, sys)