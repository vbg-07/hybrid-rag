import urllib.request
import zipfile
import os

url = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scifact.zip"
zip_path = "scifact.zip"

print("📥 Downloading official BEIR SciFact test dataset (this might take a minute)...")
urllib.request.urlretrieve(url, zip_path)

print("📦 Extracting files...")
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(".")

# Clean up the zip file
os.remove(zip_path)

print("✅ Success! You now have a 'scifact' folder containing queries.jsonl and qrels/test.tsv.")