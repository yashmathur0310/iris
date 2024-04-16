# app.py
from flask import Flask, render_template, request
import numpy as np
import seaborn as sns 
from sklearn.linear_model import LogisticRegression

app = Flask(__name__)

# Load dataset
df = sns.load_dataset("iris")
x = df.iloc[:, :-1].values
y = df['species'].values

# Train the model
model = LogisticRegression()
model.fit(x, y)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get form data
    sepal_length = float(request.form['sepal_length'])
    sepal_width = float(request.form['sepal_width'])
    petal_length = float(request.form['petal_length'])
    petal_width = float(request.form['petal_width'])

    # Make prediction
    input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    prediction = model.predict(input_data)

    # Return prediction result
    return f"The predicted species is: {prediction[0]}"

if __name__ == '__main__':
    app.run(debug=True)
