import os

import joblib
import numpy as np
from flask import Flask, render_template, request, session

app = Flask(__name__)
app.secret_key = 'used-bike-price-prediction'

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# The app uses ml_used_bikes.lb — the Linear Regression model trained in
# ml_used_bikes.ipynb on the bundled dataset (test R2 = 0.70).
# The older model.lb (Random Forest) belongs to a different dataset
# version: scored on this dataset it gives R2 < 0, so it is not used.
model = joblib.load(os.path.join(BASE_DIR, 'ml_used_bikes.lb'))

# Brand encoding exactly as produced by LabelEncoder in the data-cleaning
# step (alphabetical order). Verified against the dataset: e.g. code 17's
# bikes are 350cc / ~Rs 1.1 lakh / most common premium brand (Royal
# Enfield), code 1 is the most common 150cc commuter brand (Bajaj).
# The hand-written dictionary in the previous app version had these
# codes shuffled, so brand-specific predictions were wrong.
BRAND_CODES = {
    'BMW': 0, 'Bajaj': 1, 'Benelli': 2, 'Ducati': 3, 'Harley-Davidson': 4,
    'Hero': 5, 'Honda': 6, 'Hyosung': 7, 'Ideal': 8, 'Indian': 9,
    'Jawa': 10, 'KTM': 11, 'Kawasaki': 12, 'LML': 13, 'MV': 14,
    'Mahindra': 15, 'Rajdoot': 16, 'Royal Enfield': 17, 'Suzuki': 18,
    'TVS': 19, 'Triumph': 20, 'Yamaha': 21, 'Yezdi': 22,
}

# Feature order the model was trained on:
# [owner, brand, kilometer, age, power]
# (the previous app sent [brand, owner, age, power, kms] — wrong order)


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/contact')
def contact():
    return render_template('contact.html')


@app.route('/about')
def about():
    return render_template('about.html')


@app.route('/health')
def health():
    return {'status': 'ok'}


@app.route('/history')
def history():
    return render_template(
        'history.html', historical_data=session.get('history', []))


@app.route('/project', methods=['POST', 'GET'])
def predict():
    if request.method == 'POST':
        brand_name = request.form.get('brand_name')

        try:
            owner = int(request.form.get('owner'))
            age = int(request.form.get('age'))
            power = float(request.form.get('power'))
            kms_driven = int(request.form.get('kms_driven'))
        except (ValueError, TypeError):
            return render_template(
                'project.html',
                prediction='⚠️ Invalid input — age, power and kms must be numbers.')

        if brand_name not in BRAND_CODES:
            return render_template(
                'project.html', prediction='⚠️ Please select a valid brand.')
        if not (1 <= owner <= 4):
            return render_template(
                'project.html', prediction='⚠️ Owner must be between 1 and 4.')
        if not (0 <= age <= 40):
            return render_template(
                'project.html', prediction='⚠️ Bike age must be between 0 and 40 years.')
        if not (50 <= power <= 2000):
            return render_template(
                'project.html', prediction='⚠️ Power must be between 50 and 2000 cc.')
        if not (0 <= kms_driven <= 1000000):
            return render_template(
                'project.html', prediction='⚠️ Kms driven looks out of range.')

        input_vector = [owner, BRAND_CODES[brand_name], kms_driven, age, power]
        # model was trained with y as a one-column DataFrame, so predict()
        # returns a 2-D array — flatten before converting to float
        pred = float(np.asarray(model.predict([input_vector])).ravel()[0])
        pred = max(0.0, round(pred, 2))

        entries = session.get('history', [])
        entries.insert(0, (pred, owner, brand_name, kms_driven, age, power))
        session['history'] = entries[:10]

        return render_template(
            'project.html', prediction=pred,
            detail={'brand': brand_name, 'owner': owner, 'age': age,
                    'power': power, 'kms': kms_driven})

    return render_template('project.html', prediction=None)


if __name__ == '__main__':
    app.run(debug=True)
