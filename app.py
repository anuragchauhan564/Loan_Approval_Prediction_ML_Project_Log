import pickle 
import numpy as np
import pandas as pd
from flask import Flask,request,jsonify,render_template
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

## iporting ridge regresor model and standard scaler pickle

ridge_model = pickle.load(open('models/modelForPredictionLoan.pkl','rb'))
standard_scaler = pickle.load(open('models/standardscaler.pkl','rb'))

# Route for home page
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predictdata',methods=['GET','POST'])
def predict_datapoints():
    if request.method == 'POST':
        no_of_dependents = float(request.form.get('no_of_dependents'))
        education = float(request.form.get('education'))
        self_employed = float(request.form.get('self_employed'))
        income_annum = float(request.form.get('income_annum'))
        loan_amount = float(request.form.get('loan_amount'))
        loan_term = float(request.form.get('loan_term'))
        cibil_score = float(request.form.get('cibil_score'))
        residential_assets_value = float(request.form.get('residential_assets_value'))
        commercial_assets_value = float(request.form.get('commercial_assets_value'))
        luxury_assets_value = float(request.form.get('luxury_assets_value'))
        bank_asset_value = float(request.form.get('bank_asset_value'))

        new_data_scaled = standard_scaler.transform([[no_of_dependents,education,self_employed,income_annum,loan_amount,loan_term,cibil_score,residential_assets_value,commercial_assets_value,luxury_assets_value,bank_asset_value]])
        
        result = ridge_model.predict(new_data_scaled)
        if result == 1:
            return render_template('home.html',result = 'Approve')
        else:
            return render_template('home.html',result = 'Not Approve')
    else:
        return render_template('home.html')
    
if __name__=="__main__":
    app.run(host="0.0.0.0")
