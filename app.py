from flask import Flask, render_template, request
import joblib
import numpy as np
import datetime
import time

app = Flask(__name__)
model = joblib.load('earthquake_rf_model.joblib')

def create_timestamp(date_str, time_str):
    try:
        ts = datetime.datetime.strptime(date_str + ' ' + time_str, '%m/%d/%Y %H:%M:%S')
        return time.mktime(ts.timetuple())
    except (ValueError, OverflowError):
        return 0

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    error = None
    if request.method == 'POST':
        date = request.form.get('date')
        t = request.form.get('time')
        lat = request.form.get('latitude')
        lon = request.form.get('longitude')
        try:
            timestamp = create_timestamp(date, t)
            lat = float(lat)
            lon = float(lon)
            X_input = np.array([[timestamp, lat, lon]])
            pred = model.predict(X_input)
            mag = round(pred[0][0], 3)
            dep = round(pred[0][1], 3)
            # Damage analysis logic
            if mag < 4.0:
                damage = "Little to no damage expected."
            elif mag < 5.0:
                damage = "Minor damage possible, especially to poorly built structures."
            elif mag < 6.0:
                damage = "Moderate damage possible near the epicenter."
            elif mag < 7.0:
                damage = "Significant damage likely in affected areas."
            else:
                damage = "Severe damage expected, potentially catastrophic."
            if dep < 70:
                depth_comment = "Shallow earthquake: more likely to cause surface damage."
            elif dep < 300:
                depth_comment = "Intermediate depth: some potential for surface damage."
            else:
                depth_comment = "Deep earthquake: less likely to cause severe surface damage."
            prediction = {
                'Magnitude': mag,
                'Depth': dep,
                'Damage': damage,
                'DepthComment': depth_comment
            }
        except Exception as e:
            error = f"Invalid input or prediction error: {str(e)}"
    return render_template('index.html', prediction=prediction, error=error)

if __name__ == '__main__':
    app.run(debug=True)
