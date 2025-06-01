import zipfile
import os

zip_path = r"C:\Users\umaam\Downloads\archive.zip"
extract_to = r"C:\Users\umaam\Downloads\archive"

# Extract zip file
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(extract_to)

print("✅ Extracted successfully!")
