# pyrefly: ignore [missing-import]
from pred_pipeline import CustomData
from flask import flask,request,render_template
from src.pepelines.pred_pipeline import CustomData,Pred_Pipelines

# pyrefly: ignore [missing-import]
import numpy as np 
import pandas as pd 

from sklearn.preprocessing import StandardScaler

application=flask(__main__)

app=application

@app.route('/')
def main():
    render_template('main.py')

@app.route('/predictData',methods=["GET","POST"])
def predict_datapoint():
    
    if request=='POST':
        pass

    if request=='GET':
        return CustomData(
            keyword=request.form.get('keywords'),
            experiencelevel=request.form.get('experiencelevel')
        )
        pred_df=data.get_data_frame()
        
        predict_pipelines=Pred_Pipelines()
        predict_pipelines.predict(pred_df)
    
        return result[0]

