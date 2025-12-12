from flask import Flask, render_template, request
import numpy as np
import joblib

app = Flask(__name__)

# Load trained model
model = joblib.load("tips_model.pkl")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        total_bill = float(request.form['total_bill'])
        sex = int(request.form['sex'])         # 0 = Female, 1 = Male
        smoker = int(request.form['smoker'])   # 0 = No, 1 = Yes
        day = int(request.form['day'])         # 0=Thur, 1=Fri, 2=Sat, 3=Sun
        time = int(request.form['time'])       # 0=Lunch, 1=Dinner
        size = int(request.form['size'])

        # Arrange features in EXACT same order used in training
        features = np.array([[total_bill, sex, smoker, day, time, size]])

        prediction = model.predict(features)[0]

        return render_template("result.html", prediction=round(prediction, 2))

    except Exception as e:
        return f"Error: {str(e)}"


if __name__ == '__main__':
    app.run(debug=True)
