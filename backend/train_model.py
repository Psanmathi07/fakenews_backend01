import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import joblib
import os

print("📥 Loading dataset...")

# Load dataset
fake = pd.read_csv("data/Fake.csv")
true = pd.read_csv("data/True.csv")

# Add labels (0 = Fake, 1 = Real)
fake["label"] = 0
true["label"] = 1

# Combine and shuffle
df = pd.concat([fake, true], axis=0).sample(frac=1, random_state=42)

# Split features and labels
X = df["text"]
y = df["label"]

print("🔄 Vectorizing...")
vectorizer = TfidfVectorizer(stop_words="english", max_df=0.7)
X = vectorizer.fit_transform(X)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("🤖 Training model...")
model = LogisticRegression(max_iter=500)
model.fit(X_train, y_train)

# Evaluate
acc = model.score(X_test, y_test)
print(f"✅ Model trained with accuracy: {acc:.2f}")

# Save model and vectorizer
joblib.dump(model, "backend/model.joblib")
joblib.dump(vectorizer, "backend/vectorizer.joblib")

print("💾 Model + vectorizer saved!")
