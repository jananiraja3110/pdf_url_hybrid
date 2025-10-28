import os
import re
import pandas as pd
from urllib.parse import urlparse

# -----------------------------
# Feature extraction function
# -----------------------------
def extract_url_features(url):
    parsed = urlparse(url)
    features = {
        'url_length': len(url),
        'hostname_length': len(parsed.netloc),
        'path_length': len(parsed.path),
        'has_https': int(parsed.scheme == 'https'),
        'num_digits': len(re.findall(r'\d', url)),
        'num_special_chars': len(re.findall(r'[^\w]', url)),
        'num_subdomains': parsed.netloc.count('.'),
        'has_ip': int(bool(re.match(r'\d+\.\d+\.\d+\.\d+', parsed.netloc))),
        'has_at_symbol': int('@' in url),
        'has_hyphen': int('-' in parsed.netloc),
        'has_double_slash': int('//' in parsed.path),
        'num_parameters': len(parsed.query.split('&')) if parsed.query else 0
    }
    return features

# -----------------------------
# Build URL feature dataset
# -----------------------------
def build_url_feature_dataset():
    # Absolute path to urls folder
    base_dir = r"C:\Users\LENOVO\OneDrive\Desktop\pdf_url_hybrid\data\urls"

    # Create folder if it does not exist
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
        print(f"📂 Created folder: {base_dir}")
        print("⚠️ Please add your CSV dataset in this folder and rerun the script.")
        return

    # Find any CSV file
    csv_files = [f for f in os.listdir(base_dir) if f.endswith('.csv')]
    if not csv_files:
        print(f"⚠️ No CSV files found in {base_dir}. Please add your dataset.")
        return

    csv_path = os.path.join(base_dir, csv_files[0])
    save_path = os.path.join(base_dir, "url_features.csv")

    print(f"📥 Loading dataset: {csv_path}")
    df = pd.read_csv(csv_path)

    # Standardize column names
    df.columns = [col.strip().lower().replace(" ", "_") for col in df.columns]

    # Ensure label exists
    if 'label' not in df.columns:
        if 'label_num' in df.columns:
            df['label'] = df['label_num']
        else:
            df['label'] = df['url'].apply(lambda x: 1 if 'mal' in x else 0)

    # Extract features
    feature_list = []
    for url in df['url'].astype(str):
        feature_list.append(extract_url_features(url))

    feature_df = pd.DataFrame(feature_list)
    # Map labels to 0=benign, 1=malicious
    feature_df['label'] = df['label'].map({'benign': 0, 'malicious': 1}) if df['label'].dtype == object else df['label']

    # Save features
    feature_df.to_csv(save_path, index=False)
    print(f"✅ URL features saved to: {save_path}")

# -----------------------------
# Run script
# -----------------------------
if __name__ == "__main__":
    build_url_feature_dataset()
