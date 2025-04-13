from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import joblib
from tensorflow.keras.models import load_model
import os

app = Flask(__name__)

# Enable CORS only for the /predict route and restrict it to a specific origin (e.g., "https://yourfrontend.com")
CORS(app, resources={r"/predict": {"origins": "https://hfailure-backend-3.onrender.com"}})

# Load model and scaler
model = load_model("heart_disease_model.h5")
scaler = joblib.load("scaler.pkl")

# Encoding maps
sex_map = {'M': 1, 'F': 0}
chest_pain_map = {'ATA': 0, 'NAP': 1, 'ASY': 2, 'TA': 3}
resting_ecg_map = {'Normal': 1, 'ST': 2, 'LVH': 0}
exercise_angina_map = {'N': 0, 'Y': 1}
st_slope_map = {'Up': 2, 'Flat': 1, 'Down': 0}

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()

    try:
        # Extract and encode input features
        features = [
            data['Age'],
            sex_map[data['Sex']],
            chest_pain_map[data['ChestPainType']],
            data['RestingBP'],
            data['Cholesterol'],
            data['FastingBS'],
            resting_ecg_map[data['RestingECG']],
            data['MaxHR'],
            exercise_angina_map[data['ExerciseAngina']],
            data['Oldpeak'],
            st_slope_map[data['ST_Slope']]
        ]

        # Preprocess and scale the input features
        features = np.array([features])
        features_scaled = scaler.transform(features)

        # Make the prediction
        prediction = model.predict(features_scaled)[0][0]
        result = "Heart Disease Risk" if prediction > 0.5 else "No Risk"

        return jsonify({
            "prediction": result,
            "probability": float(prediction)
        })

    except Exception as e:
        # Enhanced error message
        return jsonify({
            "error": "An error occurred while processing your request.",
            "message": str(e)
        })

# Entry point for local and Render deployment
if __name__ == '__main__':
    # Get the port number from the environment variable or default to 10000
    port = int(os.environ.get("PORT", 10000))
    app.run(host='0.0.0.0', port=port)
