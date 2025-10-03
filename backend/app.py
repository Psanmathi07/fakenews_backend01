from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib, os

app = Flask(__name__)
CORS(app)  # ✅ allow frontend requests

model_path = os.path.join(os.path.dirname(__file__), "model.joblib")
vectorizer_path = os.path.join(os.path.dirname(__file__), "vectorizer.joblib")

try:
    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)
except Exception as e:
    model, vectorizer = None, None
    print("⚠️ Error loading model/vectorizer:", e)

@app.route("/")
def home():
    return "✅ Fake News Detector Backend running!"

@app.route("/predict", methods=["POST"])
def predict():
    if not model or not vectorizer:
        return jsonify({"error": "Model not loaded"}), 500

    data = request.get_json()
    if not data or "text" not in data:
        return jsonify({"error": "No text provided"}), 400

    text = [data["text"]]
    features = vectorizer.transform(text)
    prediction = model.predict(features)[0]

    return jsonify({"result": "Fake" if prediction == 0 else "Real"})
