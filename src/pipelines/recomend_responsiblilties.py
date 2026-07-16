

import os
import sys
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.logger import logging
from src.exception import CustomeExpection


class RecommendResponsibilities:
    def __init__(self, dataset_path: str = os.path.join('artifacts', 'raw_data.csv')):
        try:
            logging.info('Loading dataset and fitting TF-IDF for responsibility recommendation')

            self.df = pd.read_csv(dataset_path)

            combined = (
                self.df['Keywords'].fillna('').astype(str) + ' ' +
                self.df['ExperienceLevel'].fillna('').astype(str) + ' ' +
                self.df['Title'].fillna('').astype(str)
            )

            self.tfidf = TfidfVectorizer(max_features=500)
            self.combined_tfidf = self.tfidf.fit_transform(combined)

        except Exception as e:
            raise CustomeExpection(e, sys)

    def recommend(self, keyword: str, experiencelevel: str, predicted_title: str) -> str:
        try:
            user_text = f"{keyword} {experiencelevel} {predicted_title}"


            user_tfidf = self.tfidf.transform([user_text])

            similarity = cosine_similarity(self.combined_tfidf, user_tfidf)
            best_index = similarity[:, 0].argmax()

            return self.df.iloc[best_index]['Responsibilities']

        except Exception as e:
            raise CustomeExpection(e, sys)