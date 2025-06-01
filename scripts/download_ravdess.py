import os
import zipfile
from kaggle.api.kaggle_api_extended import KaggleApi

# Step 1: Set Kaggle JSON location
os.environ['KAGGLE_CONFIG_DIR'] = os.path.join(os.getcwd(), '.kaggle')

# Step 2: Initialize API and authenticate
api = KaggleApi()
api.authenticate()

# Step 3: Download dataset
print("[INFO] Downloading RAVDESS dataset...")
api.dataset_download_files(
    'uwrfkaggler/ravdess-emotional-speech-audio', path='data/', unzip=True)
print("[INFO] Dataset downloaded and extracted to /data/")
