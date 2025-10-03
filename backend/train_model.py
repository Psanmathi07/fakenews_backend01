import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import joblib
import subprocess

DATA_DIR = "data"
FAKE_PATH = os.path.join(DATA_DIR, "Fake.csv")
TRUE_PATH = os.path.join(DATA_DIR, "True.csv")

print("📥 Checking dataset...")

# Step 1: Download dataset if not present
if not (os.path.exists(FAKE_PATH) and os.path.exists(TRUE_PATH)):
    print("⚠️ Dataset not found. Downloading from Kaggle...")
    os.makedirs(DATA_DIR, exist_ok=True)
    subprocess.run([
        "kaggle", "datasets", "download", "-d",
        "clmentbisaillon/fake-and-real-news-dataset", "-p", DATA_DIR, "--unzip"
    ], check=True)

# Step 2: Load dataset
print("📥 Loading dataset...")
fake = pd.read_csv(FAKE_PATH)
true = pd.read_csv(TRUE_PATH)

fake["label"] = 0  # Fake
true["label"] = 1  # Real

df = pd.concat([fake, true], axis=0).sample(frac=1, random_state=42)

X = df["text"]
y = df["label"]

# Step 3: Vectorize
print("🔄 Vectorizing...")
vectorizer = TfidfVectorizer(stop_words="english", max_df=0.7)
X = vectorizer.fit_transform(X)

# Step 4: Train model
print("🤖 Training model...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression(max_iter=500)
model.fit(X_train, y_train)

acc = model.score(X_test, y_test)
print(f"✅ Model trained with accuracy: {acc:.2f}")

# Step 5: Save model + vectorizer
os.makedirs("backend", exist_ok=True)
joblib.dump(model, "backend/model.joblib")
joblib.dump(vectorizer, "backend/vectorizer.joblib")

print("💾 Model + vectorizer saved successfully!")
