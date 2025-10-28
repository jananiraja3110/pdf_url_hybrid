import os
import pandas as pd
import numpy as np
from flask import Flask, render_template, request, send_file, redirect, url_for
import joblib

app = Flask(__name__)
app.secret_key = "supersecretkey"

# ---------------------------------------------
# Paths
# ---------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")

# Load hybrid model pickle
pkl_files = [f for f in os.listdir(MODELS_DIR)
             if f.endswith(".pkl") and "feature_names" not in f.lower()]
if not pkl_files:
    raise FileNotFoundError(f"No hybrid model .pkl file found in {MODELS_DIR}")
MODEL_PATH = os.path.join(MODELS_DIR, pkl_files[0])
print(f"Loading model from: {MODEL_PATH}")

saved = joblib.load(MODEL_PATH)
if "model" not in saved or "scaler" not in saved:
    raise KeyError(f"The .pkl file must contain 'model' and 'scaler'. Found keys: {list(saved.keys())}")

hybrid_model = saved["model"]
scaler = saved["scaler"]
feature_names = saved.get("feature_names")  # can be dict or list

# ---------------------------------------------
# Handle feature_names structure
# ---------------------------------------------
if isinstance(feature_names, dict):
    pdf_features = feature_names['pdf_features']
    url_features = feature_names['url_features']
elif isinstance(feature_names, list):
    # If list, split manually
    PDF_FEATURE_COUNT = 19  # set according to your model
    pdf_features = feature_names[:PDF_FEATURE_COUNT]
    url_features = feature_names[PDF_FEATURE_COUNT:]
else:
    raise TypeError("feature_names must be a dict or list")

pdf_expected_len = len(pdf_features)
url_expected_len = len(url_features)
total_features = pdf_expected_len + url_expected_len

# Store last batch results
last_batch_results = None

# ---------------------------------------------
# Helper Functions
# ---------------------------------------------
def preprocess_single(pdf_feat, url_feat):
    # Auto-fill missing features with 0
    if len(pdf_feat) < pdf_expected_len:
        pdf_feat += [0]*(pdf_expected_len - len(pdf_feat))
    if len(url_feat) < url_expected_len:
        url_feat += [0]*(url_expected_len - len(url_feat))
    combined = np.array(pdf_feat + url_feat, dtype=float).reshape(1, -1)
    return scaler.transform(combined)

def preprocess_csv(pdf_df, url_df):
    # Add missing columns with 0
    for col in pdf_features:
        if col not in pdf_df.columns:
            pdf_df[col] = 0
    for col in url_features:
        if col not in url_df.columns:
            url_df[col] = 0

    # Keep only expected columns
    pdf_df = pdf_df[pdf_features]
    url_df = url_df[url_features]

    combined = np.hstack([pdf_df.values, url_df.values])
    return scaler.transform(combined)

def get_prediction_label(pred_value):
    return "Malicious" if pred_value > 0.5 else "Benign"

# ---------------------------------------------
# Routes
# ---------------------------------------------
@app.route("/", methods=["GET", "POST"])
def index():
    global last_batch_results
    result = {}
    download_available = False

    if request.method == "POST":
        # --- Single prediction ---
        if "pdf_features" in request.form and "url_features" in request.form:
            try:
                pdf_input = [float(x.strip()) for x in request.form["pdf_features"].split(",") if x.strip() != ""]
                url_input = [float(x.strip()) for x in request.form["url_features"].split(",") if x.strip() != ""]
                X = preprocess_single(pdf_input, url_input)
                preds = hybrid_model.predict(X)
                pred_value = preds[0][0] if hasattr(preds[0], "__len__") else preds[0]
                result[0] = get_prediction_label(pred_value)
            except Exception as e:
                result[0] = f"Error: {str(e)}"

        # --- Batch CSV prediction ---
        elif "pdf_file" in request.files and "url_file" in request.files:
            pdf_file = request.files["pdf_file"]
            url_file = request.files["url_file"]
            if pdf_file.filename == "" or url_file.filename == "":
                result[0] = "Error: CSV file missing"
            else:
                try:
                    pdf_df = pd.read_csv(pdf_file)
                    url_df = pd.read_csv(url_file)
                    X = preprocess_csv(pdf_df, url_df)
                    preds = hybrid_model.predict(X)

                    for idx, p in enumerate(preds):
                        value = p[0] if hasattr(p, "__len__") else p
                        result[idx] = get_prediction_label(value)

                    last_batch_results = pd.DataFrame({
                        "Index": list(result.keys()),
                        "Prediction": list(result.values())
                    })
                    download_available = True

                except Exception as e:
                    result[0] = f"Error: {str(e)}"

    # Pie chart
    valid_preds = [v for v in result.values() if v in ["Benign", "Malicious"]]
    benign_count = valid_preds.count("Benign")
    malicious_count = valid_preds.count("Malicious")
    pie_chart = None
    if valid_preds:
        pie_chart = {
            "labels": ["Benign", "Malicious"],
            "data": [benign_count, malicious_count],
            "colors": ["#28a745", "#dc3545"]
        }

    return render_template(
        "index.html",
        result=result,
        download_available=download_available,
        pie_chart=pie_chart
    )

# --- Download route ---
@app.route("/download")
def download():
    global last_batch_results
    if last_batch_results is None:
        return redirect(url_for("index"))
    path = os.path.join(BASE_DIR, "batch_predictions.csv")
    last_batch_results.to_csv(path, index=False)
    return send_file(path, as_attachment=True)

# ---------------------------------------------
# Run App
# ---------------------------------------------
if __name__ == "__main__":
    app.run(debug=True)
