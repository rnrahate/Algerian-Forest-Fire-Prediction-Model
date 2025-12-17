import pickle
from flask import Flask, request, jsonify, render_template
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib

application = Flask(__name__)
app = application

# loading model and scaler code remains unchanged
ridge_model = joblib.load('models/best_model.pkl')
standard_scaler = joblib.load('models/ridge_scaler.pkl')

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predictdata", methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'POST':
        Temperature = float(request.form['Temperature'])
        RH = float(request.form['RH'])
        Ws = float(request.form['Ws'])
        Rain = float(request.form['Rain'])
        FFMC = float(request.form['FFMC'])
        DMC = float(request.form['DMC'])
        ISI = float(request.form['ISI'])
        Classes = float(request.form['Classes'])
        Region = float(request.form['Region'])

        new_data_scaled = standard_scaler.transform([[Temperature, RH, Ws, Rain, FFMC, DMC, ISI, Classes, Region]])
        result = ridge_model.predict(new_data_scaled)

        return render_template('home.html',result = result[0])
    
    else:
        return render_template('home.html')

if __name__ == "__main__":
    app.run(host="0.0.0.0")
