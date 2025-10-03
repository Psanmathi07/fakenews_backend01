from flask import Flask, request, jsonify
import joblib
import os

app = Flask(__name__)

MODEL_PATH = "backend/model.joblib"
VECTORIZER_PATH = "backend/vectorizer.joblib"

# Load model & vectorizer
model, vectorizer = None, None
if os.path.exists(MODEL_PATH) and os.path.exists(VECTORIZER_PATH):
    try:
        model = joblib.load(MODEL_PATH)
        vectorizer = joblib.load(VECTORIZER_PATH)
        print("✅ Model and vectorizer loaded successfully.")
    except Exception as e:
        print(f"⚠️ Error loading model/vectorizer: {e}")
else:
    print("❌ Model or vectorizer not found. Please train first.")


@app.route("/")
def home():
    return jsonify({"message": "✅ Fake News Detection API is running"})


@app.route("/predict", methods=["POST"])
def predict():
    global model, vectorizer

    if not model or not vectorizer:
        return jsonify({"error": "Model not loaded"}), 500

    data = request.get_json()
    if not data or "text" not in data:
        return jsonify({"error": "Please provide 'text' field"}), 400

    text = [data["text"]]
    text_vectorized = vectorizer.transform(text)
    prediction = model.predict(text_vectorized)[0]

    result = "Real" if prediction == 1 else "Fake"

    return jsonify({"prediction": result})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
