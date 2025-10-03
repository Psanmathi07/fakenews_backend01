from flask import Flask, request, jsonify
import joblib
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Load model + vectorizer
try:
    model = joblib.load("backend/model.joblib")
    vectorizer = joblib.load("backend/vectorizer.joblib")
except Exception as e:
    model, vectorizer = None, None
    print(f"❌ Error loading model/vectorizer: {e}")

@app.route("/predict", methods=["POST"])
def predict():
    if not model or not vectorizer:
        return jsonify({"error": "Model or vectorizer not found"}), 500

    data = request.get_json()
    if "text" not in data:
        return jsonify({"error": "No text provided"}), 400

    text = data["text"]
    X = vectorizer.transform([text])
    prediction = model.predict(X)[0]
    proba = model.predict_proba(X).max()

    return jsonify({
        "prediction": "FAKE" if prediction == 0 else "REAL",
        "confidence": round(float(proba) * 100, 2)
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
