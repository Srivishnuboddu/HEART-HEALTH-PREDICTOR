# app.py
from flask import Flask, render_template, request #for taking i/p data 
import pickle #This module is used to load the machine learning model
import pandas as pd

app = Flask(__name__)

# Load the trained model
model_path = 'model.pkl'
with open(model_path, 'rb') as file:
    model = pickle.load(file)  

# Define the route for the home page
@app.route('/')
def home():
    return render_template('index.html')

# Define the route for the prediction page
@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        # Retrieve the form data
        age = int(request.form['age'])
        restbp = float(request.form['restbp'])
        chol = float(request.form['chol'])
        maxhr = float(request.form['maxhr'])
        oldpeak = float(request.form['oldpeak'])
        slope = int(request.form['slope'])
        ca = int(request.form['ca'])
        thal = request.form['thal']

        # Create a DataFrame for model prediction
        input_data = pd.DataFrame([[age, restbp, chol, maxhr, oldpeak, slope, ca, thal]],
                                  columns=['Age', 'RestBP', 'Chol', 'MaxHR', 'Oldpeak', 'Slope', 'Ca', 'Thal'])

        # Encode the input data if necessary
        input_data = pd.get_dummies(input_data, columns=['Thal', 'Slope'], drop_first=True)

        # Align columns to match the model's input
        model_columns = list(model.feature_names_in_)
        input_data = input_data.reindex(columns=model_columns, fill_value=0)

        # Make prediction using the trained model
        prediction = model.predict(input_data)[0]
        result = 'Positive (Heart Disease)' if prediction == 1 else 'Negative (No Heart Disease)'

        return render_template('predict.html', prediction_text=f'Result: {result}')
    
    return render_template('predict.html')

if __name__ == '__main__':
    app.run(debug=True)
