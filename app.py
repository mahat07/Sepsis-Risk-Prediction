from flask import Flask, render_template, request
import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler

app = Flask(__name__, template_folder='templates')


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Load the trained model
    voting_classifier = joblib.load('models/voting_classifier_model.joblib')

    # Get input values from form submission
    HR_mean = float(request.form['HR_mean'])
    O2Sat_mean = float(request.form['O2Sat_mean'])
    Temp_mean = float(request.form['Temp_mean'])
    MAP_mean = float(request.form['MAP_mean'])
    Resp_mean = float(request.form['Resp_mean'])
    FiO2_mean = float(request.form['FiO2_mean'])
    SaO2_mean = float(request.form['SaO2_mean'])
    AST_mean = float(request.form['AST_mean'])
    BUN_mean = float(request.form['BUN_mean'])
    Creatinine_mean = float(request.form['Creatinine_mean'])
    Glucose_mean = float(request.form['Glucose_mean'])
    Hgb_mean = float(request.form['Hgb_mean'])
    WBC_mean = float(request.form['WBC_mean'])
    Platelets_mean = float(request.form['Platelets_mean'])
    Age_mean = float(request.form['Age_mean'])
    Gender_first = float(request.form['Gender_first'])
    HospAdmTime_first = float(request.form['HospAdmTime_first'])
    ICULOS_max = float(request.form['ICULOS_max'])



    # Preprocess the input data
    input_values = np.array([[HR_mean, O2Sat_mean, Temp_mean, MAP_mean, Resp_mean, FiO2_mean, SaO2_mean, AST_mean,
                              BUN_mean, Creatinine_mean, Glucose_mean, Hgb_mean, WBC_mean, Platelets_mean, Age_mean,
                              Gender_first, HospAdmTime_first, ICULOS_max]])
    scaler = StandardScaler()
    input_scaled = scaler.fit_transform(input_values)

    # Make predictions
    predicted_sepsis_risk = voting_classifier.predict_proba(input_scaled)[:, 1]  # Probability of sepsis

    return render_template('result.html', sepsis_risk=predicted_sepsis_risk)

if __name__ == '__main__':
    app.run(debug=True)
