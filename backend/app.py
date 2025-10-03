# backend/app.py
import joblib
from flask import Flask, request, jsonify
import os

app = Flask(__name__)

model_path = os.path.join(os.path.dirname(__file__), "model.joblib")
vectorizer_path = os.path.join(os.path.dirname(__file__), "vectorizer.joblib")

try:
    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)
except Exception as e:
    model, vectorizer = None, None
    print("❌ Error loading model/vectorizer:", e)

@app.route("/predict", methods=["POST"])
def predict():
    if model is None or vectorizer is None:
        return jsonify({"error": "Model not loaded on server"}), 500
    text = request.json.get("text")
    X = vectorizer.transform([text])
    pred = model.predict(X)[0]
    label = "Real" if pred == 1 else "Fake"
    return jsonify({"prediction": label})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
