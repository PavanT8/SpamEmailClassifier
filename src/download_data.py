import urllib.request
import zipfile
import os

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip"
data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "raw")
zip_path = os.path.join(data_dir, "smsspamcollection.zip")

print("Downloading dataset...")
urllib.request.urlretrieve(url, zip_path)
print("Download complete.")

print("Extracting dataset...")
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(data_dir)
print("Extraction complete.")

# Clean up zip file
os.remove(zip_path)
print("Dataset downloaded and ready in data/raw/")
