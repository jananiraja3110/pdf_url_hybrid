import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib

# -----------------------------
# Load URL features
# -----------------------------
base_dir = r"C:\Users\LENOVO\OneDrive\Desktop\pdf_url_hybrid\data\urls"
features_file = os.path.join(base_dir, "url_features.csv")

if not os.path.exists(features_file):
    raise FileNotFoundError(f"URL features file not found: {features_file}")

df = pd.read_csv(features_file)

# -----------------------------
# Prepare data
# -----------------------------
X = df.drop(columns=['label'])
y = df['label']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# -----------------------------
# Train RandomForest classifier
# -----------------------------
clf = RandomForestClassifier(
    n_estimators=200,
    max_depth=None,
    random_state=42,
    n_jobs=-1
)
clf.fit(X_train, y_train)

# -----------------------------
# Evaluate model
# -----------------------------
y_pred = clf.predict(X_test)

print("✅ Model trained successfully!")
print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# -----------------------------
# Save trained model
# -----------------------------
model_path = os.path.join(base_dir, "url_model.pkl")
joblib.dump(clf, model_path)
print(f"💾 Trained model saved to: {model_path}")
