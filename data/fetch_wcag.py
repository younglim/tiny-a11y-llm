import os
import subprocess
import glob
from datasets import Dataset

DATASET_DIR = "datasets/wcag"
REPO_URL = "https://github.com/w3c/wcag.git"
OUTPUT_PATH = "datasets/wcag_prepared"

# Clone repo if not already present
if not os.path.exists(DATASET_DIR):
    os.makedirs("datasets", exist_ok=True)
    subprocess.run(["git", "clone", "--depth", "1", REPO_URL, DATASET_DIR], check=True)
else:
    print("WCAG repo already exists, skipping clone.")

# Collect Markdown files
texts = []
for path in glob.glob(f"{DATASET_DIR}/**/*.md", recursive=True):
    try:
        with open(path, "r", encoding="utf-8") as f:
            content = f.read().strip()
            if content:
                texts.append({"text": content})
    except Exception as e:
        print(f"Skipping {path}: {e}")

# Create and save dataset
if texts:
    ds = Dataset.from_list(texts)
    ds.save_to_disk(OUTPUT_PATH)
    print(f"Saved {len(texts)} documents to {OUTPUT_PATH}")
else:
    print("No markdown files found.")
