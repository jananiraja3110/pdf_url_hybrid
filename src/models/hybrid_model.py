import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score
import joblib
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, Concatenate
from tensorflow.keras.optimizers import Adam

# ---------------------------------------------
# 🧩 File Paths
# ---------------------------------------------
BASE_DIR = r"C:\Users\LENOVO\OneDrive\Desktop\pdf_url_hybrid"
PDF_FEATURES_PATH = os.path.join(BASE_DIR, "data", "pdfmal", "pdf_features.csv")
URL_FEATURES_PATH = os.path.join(BASE_DIR, "data", "urls", "url_features.csv")
HYBRID_MODEL_PATH = os.path.join(BASE_DIR, "models", "hybrid_deep_model.h5")
SCALER_PATH = os.path.join(BASE_DIR, "models", "scaler.pkl")
FEATURES_PATH = os.path.join(BASE_DIR, "models", "feature_names.pkl")

# ---------------------------------------------
# ⚙️ Load and Preprocess Function
# ---------------------------------------------
def load_and_preprocess(csv_path, label_col="class"):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"❌ CSV file not found: {csv_path}")
    df = pd.read_csv(csv_path)
    if label_col not in df.columns:
        raise ValueError(f"❌ Label column '{label_col}' not found in {csv_path}")
    df = df.dropna()
    X = df.drop(columns=[label_col])
    y = df[label_col]
    if y.dtype == 'object':
        y = y.map({"Benign": 0, "Malicious": 1}).fillna(y).astype(int)
    X = X.select_dtypes(include=['number']).fillna(0)
    return X, y

# ---------------------------------------------
# 🧠 Train Hybrid Deep Learning Model
# ---------------------------------------------
def main():
    print("📄 Loading PDF data...")
    X_pdf, y_pdf = load_and_preprocess(PDF_FEATURES_PATH)

    print("🌐 Loading URL data...")
    X_url, y_url = load_and_preprocess(URL_FEATURES_PATH, label_col='label')

    # Align sizes
    min_len = min(len(X_pdf), len(X_url))
    X_pdf = X_pdf.iloc[:min_len].reset_index(drop=True)
    X_url = X_url.iloc[:min_len].reset_index(drop=True)
    y = y_pdf.iloc[:min_len]

    # Train-test split
    X_train_pdf, X_test_pdf, X_train_url, X_test_url, y_train, y_test = train_test_split(
        X_pdf, X_url, y, test_size=0.2, random_state=42, stratify=y
    )

    # Scale features
    scaler_pdf = StandardScaler()
    scaler_url = StandardScaler()
    X_train_pdf_scaled = scaler_pdf.fit_transform(X_train_pdf)
    X_train_url_scaled = scaler_url.fit_transform(X_train_url)
    X_test_pdf_scaled = scaler_pdf.transform(X_test_pdf)
    X_test_url_scaled = scaler_url.transform(X_test_url)

    # ---------------------------------------------
    # Build Hybrid Deep Learning Model
    # ---------------------------------------------
    input_pdf = Input(shape=(X_train_pdf_scaled.shape[1],), name='pdf_input')
    input_url = Input(shape=(X_train_url_scaled.shape[1],), name='url_input')

    # PDF branch
    x1 = Dense(128, activation='relu')(input_pdf)
    x1 = Dropout(0.3)(x1)
    x1 = Dense(64, activation='relu')(x1)

    # URL branch
    x2 = Dense(128, activation='relu')(input_url)
    x2 = Dropout(0.3)(x2)
    x2 = Dense(64, activation='relu')(x2)

    # Concatenate
    combined = Concatenate()([x1, x2])
    z = Dense(64, activation='relu')(combined)
    z = Dropout(0.3)(z)
    output = Dense(1, activation='sigmoid')(z)

    model = Model(inputs=[input_pdf, input_url], outputs=output)
    model.compile(optimizer=Adam(0.001), loss='binary_crossentropy', metrics=['accuracy'])
    model.summary()

    # ---------------------------------------------
    # Train model
    # ---------------------------------------------
    print("🚀 Training Hybrid Deep Learning Model...")
    history = model.fit(
        [X_train_pdf_scaled, X_train_url_scaled],
        y_train,
        validation_split=0.2,
        epochs=50,
        batch_size=32,
        verbose=1
    )

    # ---------------------------------------------
    # Evaluate
    # ---------------------------------------------
    y_pred_prob = model.predict([X_test_pdf_scaled, X_test_url_scaled])
    y_pred = (y_pred_prob > 0.5).astype(int)
    acc = accuracy_score(y_test, y_pred)
    print(f"\n✅ Hybrid Deep Learning Model Accuracy: {acc:.4f}\n")
    print(classification_report(y_test, y_pred, target_names=["Benign", "Malicious"]))

    # ---------------------------------------------
    # Save model, scalers, feature info
    # ---------------------------------------------
    os.makedirs(os.path.dirname(HYBRID_MODEL_PATH), exist_ok=True)
    model.save(HYBRID_MODEL_PATH)

    # Save scalers and feature names for Flask inference
    joblib.dump(
        {"pdf": scaler_pdf, "url": scaler_url},
        SCALER_PATH
    )
    joblib.dump(
        {"pdf_features": X_pdf.columns.tolist(), "url_features": X_url.columns.tolist()},
        FEATURES_PATH
    )

    print(f"\n💾 Hybrid deep learning model saved to: {HYBRID_MODEL_PATH}")
    print(f"💾 Scalers saved to: {SCALER_PATH}")
    print(f"💾 Feature names saved to: {FEATURES_PATH}")

if __name__ == "__main__":
    main()
