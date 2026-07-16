# import sys
# import os
# import pandas as pd

# from src.exception import CustomeExpection
# from src.logger import logging
# from src.utils import load_object


# class Pred_Pipelines:
#     def __init__(self):
#         try:
#             model_path = os.path.join('artifacts', 'model.pkl')
#             preprocessor_path = os.path.join('artifacts', 'preprocessor.pkl')
#             target_encoder_path = os.path.join('artifacts', 'target_encoder.pkl')

#             self.model = load_object(file_path=model_path)
#             self.preprocessor = load_object(file_path=preprocessor_path)
#             self.target_encoder = load_object(file_path=target_encoder_path)

#         except Exception as e:
#             raise CustomeExpection(e, sys)

#     def predict(self, features: pd.DataFrame):
#         try:
#             data_scaled = self.preprocessor.transform(features)
#             prediction_encoded = self.model.predict(data_scaled)

#             # model outputs a LabelEncoder-encoded integer; convert it back
#             # to the actual job title string using the saved target_encoder
#             prediction = self.target_encoder.inverse_transform(prediction_encoded)
#             return prediction

#         except Exception as e:
#             raise CustomeExpection(e, sys)


# class CustomData:
#     def __init__(self, keyword: str, experiencelevel: str):
#         self.keyword = keyword
#         self.experiencelevel = experiencelevel

#     def get_data_frame(self):
#         try:
#             logging.info('User input is being converted into a dataframe')

#             # training data used ';' as the keyword separator; the UI
#             # collects comma-separated keywords, so normalize here
#             normalized_keywords = self.keyword.replace(',', ';')

#             # column names must match EXACTLY what the ColumnTransformer
#             # was fit on: cat_columns=['Keywords', 'ExperienceLevel']
#             custom_data_frame = {
#                 'Keywords': [normalized_keywords],
#                 'ExperienceLevel': [self.experiencelevel]
#             }
#             return pd.DataFrame(custom_data_frame)

#         except Exception as e:
#             raise CustomeExpection(e, sys)


import sys
import os
import pandas as pd

from src.exception import CustomeExpection
from src.logger import logging
from src.utils import load_object


class Pred_Pipelines:
    def __init__(self):
        try:
            model_path = os.path.join('artifacts', 'model.pkl')
            preprocessor_path = os.path.join('artifacts', 'preprocessor.pkl')
            target_encoder_path = os.path.join('artifacts', 'target_encoder.pkl')

            self.model = load_object(file_path=model_path)
            self.preprocessor = load_object(file_path=preprocessor_path)
            self.target_encoder = load_object(file_path=target_encoder_path)

        except Exception as e:
            raise CustomeExpection(e, sys)

    def predict(self, features: pd.DataFrame):
        try:
            data_scaled = self.preprocessor.transform(features)
            prediction_encoded = self.model.predict(data_scaled)

            # train_arr/test_arr were built with np.c_[features, target],
            # which upcasts the integer target labels to float64 (since the
            # whole array must share one dtype). The model was therefore
            # trained on float labels and predicts floats back -- cast to
            # int before inverse_transform, which requires integer indices.
            prediction_encoded = prediction_encoded.astype(int)

            # model outputs a LabelEncoder-encoded integer; convert it back
            # to the actual job title string using the saved target_encoder
            prediction = self.target_encoder.inverse_transform(prediction_encoded)
            return prediction

        except Exception as e:
            raise CustomeExpection(e, sys)


class CustomData:
    def __init__(self, keyword: str, experiencelevel: str):
        self.keyword = keyword
        self.experiencelevel = experiencelevel

    def get_data_frame(self):
        try:
            logging.info('User input is being converted into a dataframe')

            # training data used ';' as the keyword separator; the UI
            # collects comma-separated keywords, so normalize here
            normalized_keywords = self.keyword.replace(',', ';')

            # column names must match EXACTLY what the ColumnTransformer
            # was fit on: cat_columns=['Keywords', 'ExperienceLevel']
            custom_data_frame = {
                'Keywords': [normalized_keywords],
                'ExperienceLevel': [self.experiencelevel]
            }
            return pd.DataFrame(custom_data_frame)

        except Exception as e:
            raise CustomeExpection(e, sys)