## Same as app.py for deployment in AWS Elasticbeanstalk
## When deploying remove the app.py file from github

from flask import Flask, request, render_template
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import CustomData, PredictionPipeline

application = Flask(__name__)

# Route for home page
@application.route('/')
def home():
    return render_template('index.html')

@application.route('/prediction', methods = ['GET', 'POST'])
def prediction():
    if request.method == 'GET':
        return render_template('home.html')
    else:
        data = CustomData(
            gender=request.form.get('gender'),
            race_ethnicity=request.form.get('race_ethnicity'),
            parental_level_of_education=request.form.get('parental_level_of_education'),
            lunch=request.form.get('lunch'),
            test_preparation_course=request.form.get('test_preparation_course'),
            reading_score=request.form.get('reading_score'),
            writing_score=request.form.get('writing_score'),
        )
        df = data.get_data_as_dataframe()

        pred_pipeline = PredictionPipeline()
        pred = pred_pipeline.predict(df)
        return render_template('home.html', results = pred[0])
    

if __name__=='__main__':
    application.run(host="0.0.0.0")