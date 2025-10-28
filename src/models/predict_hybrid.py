import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

CSV_PATH = r"C:\Users\LENOVO\OneDrive\Desktop\pdf_url_hybrid\data\pdfmal\pdf_features.csv"
MODEL_PATH = r"C:\Users\LENOVO\OneDrive\Desktop\pdf_url_hybrid\data\pdfmal\pdf_model.pkl"

# Load CSV
df = pd.read_csv(CSV_PATH)

# Detect label column
label_col = 'class' if 'class' in df.columns else 'label'

# Drop rows with missing labels
df = df.dropna(subset=[label_col])

# Split features and target
X = df.drop(columns=[label_col, 'name', 'contains_text', 'header'], errors='ignore')
y = df[label_col].copy()

# Keep only numeric columns
X = X.select_dtypes(include=['float64', 'int64'])

# Impute missing values
imputer = SimpleImputer(strategy='median')
X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_imputed, y, test_size=0.2, random_state=42)

# Train RandomForest
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Evaluate
y_pred = clf.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Save model
joblib.dump(clf, MODEL_PATH)
print(f"💾 Trained PDF model saved to: {MODEL_PATH}")
