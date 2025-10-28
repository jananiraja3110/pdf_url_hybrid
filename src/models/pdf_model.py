# pdf_model.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report
import joblib

# === CONFIG: update these paths if needed ===
CSV_PATH = r"C:\Users\LENOVO\OneDrive\Desktop\pdf_url_hybrid\data\pdfmal\pdf_features.csv"
MODEL_PATH = r"C:\Users\LENOVO\OneDrive\Desktop\pdf_url_hybrid\data\pdfmal\pdf_model.pkl"

# === LOAD DATA ===
def load_df(path):
    try:
        df = pd.read_csv(path)
        print(f"Loaded CSV with shape: {df.shape}")
        return df
    except FileNotFoundError:
        raise FileNotFoundError(f"CSV not found at: {path}")

# === FEATURE SELECTION ===
def auto_select_features(df, label_col='class'):
    if label_col not in df.columns:
        raise ValueError(f"Label column '{label_col}' not found in CSV")
    
    # Map label to 0/1 if needed
    if df[label_col].dtype == object:
        df[label_col] = df[label_col].map({'Benign': 0, 'Malicious': 1})

    # Keep numeric and boolean columns only
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    bool_cols = df.select_dtypes(include=['bool']).columns.tolist()
    drop_cols = [c for c in df.columns if c not in numeric_cols + bool_cols + [label_col]]
    
    features = numeric_cols + bool_cols
    print(f"Selected {len(features)} features ({len(numeric_cols)} numeric + {len(bool_cols)} bool-like). Dropping: {drop_cols}")
    return features

# === MAIN ===
def main():
    df = load_df(CSV_PATH)
    
    label_col = 'class'
    print("Label counts (including NaN):")
    print(df[label_col].value_counts(dropna=False))
    
    # Drop rows with invalid labels
    df = df[df[label_col].notna()]
    print(f"Shape after dropping invalid labels: {df.shape}")
    
    # Feature selection
    features = auto_select_features(df, label_col=label_col)
    X = df[features].copy()
    y = df[label_col].astype(int)
    
    # Impute missing values
    numeric_cols = X.select_dtypes(include=[np.number]).columns
    X_numeric = pd.DataFrame(SimpleImputer(strategy='median').fit_transform(X[numeric_cols]), columns=numeric_cols)
    
    bool_cols = X.select_dtypes(include=['bool']).columns
    X_bool = X[bool_cols].astype(int) if len(bool_cols) > 0 else pd.DataFrame()
    
    if not X_bool.empty:
        X_final = pd.concat([X_numeric, X_bool], axis=1)
    else:
        X_final = X_numeric
    
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X_final, y, test_size=0.2, random_state=42, stratify=y)
    
    # Train model
    print("Training RandomForest...")
    clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    clf.fit(X_train, y_train)
    
    # Evaluate
    y_pred = clf.predict(X_test)
    print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    
    # Save model
    joblib.dump(clf, MODEL_PATH)
    print(f"💾 Trained model saved to: {MODEL_PATH}")

if __name__ == "__main__":
    main()
