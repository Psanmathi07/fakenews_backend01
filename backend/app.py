import os
import joblib
from flask import Flask, request, jsonify

app = Flask(__name__)

# ✅ Load model + vectorizer safely
MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.joblib")
VECTORIZER_PATH = os.path.join(os.path.dirname(__file__), "vectorizer.joblib")

try:
    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
    print("✅ Model and Vectorizer loaded successfully.")
except Exception as e:
    print(f"⚠️ Error loading model/vectorizer: {e}")
    model, vectorizer = None, None


@app.route("/")
def home():
    return jsonify({"message": "✅ Backend running on Render!"})


@app.route("/predict", methods=["POST"])
def predict():
    if not model or not vectorizer:
        return jsonify({"error": "Model or vectorizer not loaded"}), 500

    try:
        data = request.get_json()
        text = data.get("text", "")

        if not text.strip():
            return jsonify({"error": "No text provided"}), 400

        # Transform and predict
        X = vectorizer.transform([text])
        prediction = model.predict(X)[0]
        label = "FAKE NEWS ❌" if prediction == 1 else "REAL NEWS ✅"

        return jsonify({"result": label})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    # ✅ Render requires dynamic port binding
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port, debug=False)
