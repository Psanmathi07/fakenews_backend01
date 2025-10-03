from flask import Flask, request, jsonify
import joblib
import os

app = Flask(__name__)

# ✅ Load model + vectorizer safely
MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.joblib")
VECTORIZER_PATH = os.path.join(os.path.dirname(__file__), "vectorizer.joblib")

model, vectorizer = None, None
if os.path.exists(MODEL_PATH) and os.path.exists(VECTORIZER_PATH):
    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
else:
    print("⚠️ No model found! Please train first.")

@app.route("/")
def home():
    return {"status": "Backend running ✅"}

@app.route("/predict", methods=["POST"])
def predict():
    global model, vectorizer
    if not model or not vectorizer:
        return jsonify({"error": "Model not loaded"}), 500

    data = request.get_json()
    if not data or "text" not in data:
        return jsonify({"error": "No text provided"}), 400

    text = data["text"]
    X = vectorizer.transform([text])
    pred = model.predict(X)[0]
    proba = model.predict_proba(X)[0]

    return jsonify({
        "prediction": "REAL" if pred == 1 else "FAKE",
        "confidence": round(float(max(proba)) * 100, 2)
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
