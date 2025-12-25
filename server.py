from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

model = joblib.load("fraud-detection.pkl")

THRESHOLD = 0.3

@app.route("/")
def home():
    return {"message": "Fraud Detection API running"}

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    features = np.array(data["features"]).reshape(1, -1)

    prob = model.predict_proba(features)[0][1]
    prediction = int(prob >= THRESHOLD)

    return jsonify({
        "fraud_probability": float(prob),
        "fraud_prediction": prediction
    })

app.run(host="0.0.0.0", port=5000)