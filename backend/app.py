from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import os

app = Flask(__name__)
CORS(app)  # ✅ allow frontend to call backend

# Load model + vectorizer
model_path = os.path.join(os.path.dirname(__file__), "model.joblib")
vectorizer_path = os.path.join(os.path.dirname(__file__), "vectorizer.joblib")

try:
    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)
    print("✅ Model & Vectorizer loaded successfully")
except Exception as e:
    print("⚠️ Error loading model/vectorizer:", e)
    model, vectorizer = None, None


@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "Fake News Detector Backend is running 🚀"})


@app.route("/predict", methods=["POST"])
def predict():
    if not model or not vectorizer:
        return jsonify({"error": "Model not loaded"}), 500

    data = request.get_json()
    if not data or "text" not in data:
        return jsonify({"error": "Missing 'text' in request"}), 400

    text = data["text"]
    vec = vectorizer.transform([text])
    prediction = model.predict(vec)[0]
    proba = model.predict_proba(vec)[0]

    label = "REAL" if prediction == 1 else "FAKE"
    confidence = round(max(proba) * 100, 2)

    return jsonify({"prediction": label, "confidence": confidence})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
