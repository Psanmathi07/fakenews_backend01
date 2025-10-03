import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import joblib
import os

print("📥 Loading dataset...")

# Paths (adjusted for Kaggle download)
fake_path = "data/Fake.csv"
true_path = "data/True.csv"

if not os.path.exists(fake_path) or not os.path.exists(true_path):
    raise FileNotFoundError("❌ Fake.csv or True.csv not found in data/. Make sure dataset is downloaded.")

# Load dataset
fake = pd.read_csv(fake_path)
true = pd.read_csv(true_path)

# Add labels
fake["label"] = 0  # Fake
true["label"] = 1  # Real

# Combine & shuffle
df = pd.concat([fake, true], axis=0).sample(frac=1, random_state=42)

X = df["text"]  # use text column
y = df["label"]

print("🔄 Vectorizing text...")
vectorizer = TfidfVectorizer(stop_words="english", max_df=0.7)
X = vectorizer.fit_transform(X)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("🤖 Training model...")
model = LogisticRegression(max_iter=500)
model.fit(X_train, y_train)

# Accuracy
acc = model.score(X_test, y_test)
print(f"✅ Model trained with accuracy: {acc:.2f}")

# Save model + vectorizer in backend/
os.makedirs("backend", exist_ok=True)
joblib.dump(model, "backend/model.joblib")
joblib.dump(vectorizer, "backend/vectorizer.joblib")

print("💾 Model + vectorizer saved in backend/")
