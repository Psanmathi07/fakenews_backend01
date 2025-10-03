import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import joblib
import kaggle

# Ensure data folder exists
os.makedirs("data", exist_ok=True)

print("📥 Downloading dataset from Kaggle...")

# Download Fake and Real News Dataset
# You must set KAGGLE_USERNAME and KAGGLE_KEY as GitHub/Render secrets
kaggle.api.dataset_download_files(
    "clmentbisaillon/fake-and-real-news-dataset",
    path="data",
    unzip=True
)

print("✅ Dataset downloaded & extracted!")

# Load datasets
fake = pd.read_csv("data/Fake.csv")
true = pd.read_csv("data/True.csv")

fake["label"] = 0  # Fake
true["label"] = 1  # Real

df = pd.concat([fake, true], axis=0).sample(frac=1, random_state=42)

X = df["text"]
y = df["label"]

print("🔄 Vectorizing text...")
vectorizer = TfidfVectorizer(stop_words="english", max_df=0.7)
X = vectorizer.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("🤖 Training model...")
model = LogisticRegression(max_iter=500)
model.fit(X_train, y_train)

acc = model.score(X_test, y_test)
print(f"✅ Model trained with accuracy: {acc:.2f}")

# Save model + vectorizer
joblib.dump(model, "backend/model.joblib")
joblib.dump(vectorizer, "backend/vectorizer.joblib")

print("💾 Model + vectorizer saved to backend/")
