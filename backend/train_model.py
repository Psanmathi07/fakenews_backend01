import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import joblib

print("📥 Loading dataset...")
df_fake = pd.read_csv("data/Fake.csv")
df_true = pd.read_csv("data/True.csv")

df_fake["label"] = 0
df_true["label"] = 1

df = pd.concat([df_fake, df_true], axis=0).sample(frac=1).reset_index(drop=True)

X = df["text"]
y = df["label"]

print("🔎 Vectorizing text...")
vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
X_vec = vectorizer.fit_transform(X)

print("🤖 Training model...")
model = LogisticRegression(max_iter=1000)
model.fit(X_vec, y)

print("💾 Saving model & vectorizer...")
joblib.dump(model, "backend/model.joblib")
joblib.dump(vectorizer, "backend/vectorizer.joblib")

print("✅ Training complete. Files saved in backend/")
