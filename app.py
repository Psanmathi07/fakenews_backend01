from flask import Flask, request, jsonify
import joblib
import os

# Initialize Flask app
app = Flask(__name__)

# Load model + vectorizer safely
MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.joblib")
VECTORIZER_PATH = os.path.join(os.path.dirname(__file__), "vectorizer.joblib")

try:
    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
except Exception as e:
    print("⚠️ Error loading model/vectorizer:", e)
    model = None
    vectorizer = None

@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "✅ Fake News Detection API is running!"})

@app.route("/predict", methods=["POST"])
def predict():
    if model is None or vectorizer is None:
        return jsonify({"error": "Model or vectorizer not loaded"}), 500

    data = request.get_json()
    if not data or "text" not in data:
        return jsonify({"error": "Missing 'text' in request"}), 400

    text = data["text"]

    # Transform & predict
    X = vectorizer.transform([text])
    prediction = model.predict(X)[0]
    confidence = model.predict_proba(X)[0].max() * 100

    result = "FAKE" if prediction == 0 else "REAL"
    return jsonify({
        "prediction": result,
        "confidence": round(confidence, 2)
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
