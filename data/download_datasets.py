import os
import requests
import zipfile
from tqdm import tqdm
import subprocess

# -----------------------------------------
# Directories
# -----------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
PDF_DIR = os.path.join(DATA_DIR, "pdfs")
URL_DIR = os.path.join(DATA_DIR, "urls")

for d in [DATA_DIR, PDF_DIR, URL_DIR]:
    os.makedirs(d, exist_ok=True)

# -----------------------------------------
# 1. Download Malicious URL Dataset (Kaggle)
# -----------------------------------------
def download_kaggle_dataset():
    print("\n📥 Downloading Malicious URLs Dataset from Kaggle...")
    try:
        subprocess.run([
            "kaggle", "datasets", "download",
            "-d", "sid321axn/malicious-urls-dataset",
            "-p", URL_DIR, "--unzip"
        ], check=True)
        print("✅ Malicious URL dataset downloaded and extracted.")
    except Exception as e:
        print(f"⚠️ Kaggle download failed: {e}")
        print("Please ensure the Kaggle CLI is installed and kaggle.json is configured correctly.")

# -----------------------------------------
# 2. Download CIC Evasive PDFMal2022 dataset
# -----------------------------------------
def download_pdfmal_dataset():
    print("\n📥 Downloading CIC Evasive-PDFMal2022 dataset...")
    url = "https://www.unb.ca/cic/datasets/pdfmal-2022.html"
    print(f"🔗 Visit the dataset page manually (license agreement may be required):\n{url}")
    print("After downloading the ZIP, place it in data/pdfs/ and rerun feature extraction.")
    # Automatic download often blocked by license; manual step safer.

# -----------------------------------------
# 3. Run all downloads
# -----------------------------------------
if __name__ == "__main__":
    print("🚀 Starting dataset download process...")
    download_kaggle_dataset()
    download_pdfmal_dataset()
    print("\n🎯 All downloads complete. Next step: run feature extraction.")
