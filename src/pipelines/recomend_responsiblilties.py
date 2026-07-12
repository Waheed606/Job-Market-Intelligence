
from sympy.polys.domains import characteristiczero
from sklearn.feature_extraction.text import TfidfVectorizer
from src.logger import logging
from src.exception import CustomeExpection
from src.pipelines.pred_pred_pipelines import Pred_Pipelines
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import sys 



class  recomend_responsiblilities:
    try:
        def __init__(self):
            
                predition=Pred_Pipelines()
                pred_df=pd.DataFrame(predition)
                df=pd.read_csv('job_dataset.csv')
                Combined= (df["Keywords"] +' '+df['ExperienceLevel']+' '+df["Title"])

                tfidf = TfidfVectorizer(max_features=500)
                Combined_tfidf = tfidf.fit_transform(Combined)
                pred_tfidf = tfidf.fit_transform(pred_df)

                similarity = cosine_similarity(
                Combined_tfidf,
                pred_tfidf
                )

                best_index = similarity.argmax()

                recomend_respons=df.iloc[best_index]["Responsibilities"]
                return recomend_respons
    
    except Exception as e:
        raise CustomeExpection(e,sys)


