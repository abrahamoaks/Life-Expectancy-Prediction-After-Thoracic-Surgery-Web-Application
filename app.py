import joblib
from flask import Flask, render_template, redirect, url_for, request
from flask_bootstrap import Bootstrap
from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, BooleanField
from wtforms.validators import InputRequired, Email, Length
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier




filename1 = 'C:\\Predictive models\\knn_thoracic_surg_model.pkl'
model1 = joblib.load(filename1)



app = Flask(__name__, template_folder='C:\\Users\\HP\\pythonProjects\Predio - Thoracic Surgery\\templates')


##################################################################################
    
@app.route('/')
def index():
    return render_template("thoracic.html")


@app.route("/thoracic")
def thoracic():
    return render_template("thoracic.html")

##################################################################################

@app.route('/predictsurgery', methods=['POST'])
def predictsurgery():
    if request.method == 'POST':
        # Get form inputs
        Age = float(request.form['Age'])
        Asthma = float(request.form['Asthma'])
        Pain = float(request.form['Pain'])
        Smoking = float(request.form['Smoking'])
        Cough = float(request.form['Cough'])
        Weakness = float(request.form['Weakness'])
        Performance = float(request.form['Performance'])
        Diabetes_Mellitus = float(request.form['Diabetes_Mellitus'])
        Haemoptysis = float(request.form['Haemoptysis'])
        Tumor_Size = float(request.form['Tumor_Size'])
        FVC = float(request.form['FVC'])
        FEV1 = float(request.form['FEV1'])
              

        # Create a NumPy array with the input data
        data = np.array([[Age, Asthma, Pain, Smoking, Cough, Weakness, Performance, Diabetes_Mellitus, Haemoptysis, Tumor_Size, FVC, FEV1]])
        
        # Make prediction
        my_prediction = model1.predict(data)
        
        # Convert prediction to text
        prediction_text = ""
        if my_prediction == 1:
            prediction_text = "Patient Dies After 1 Year"
        else:
            prediction_text = "Patient Lives Beyond 1 Year"


        return render_template('thoracic_result.html', prediction_text=prediction_text)

##################################################################################

if __name__ == "__main__":
    app.run(debug=True)
