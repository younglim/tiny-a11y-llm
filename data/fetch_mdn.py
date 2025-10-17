# fetch_mdn.py
import os, subprocess, glob
from datasets import Dataset

DATASET_DIR = "datasets/mdn"
OUTPUT_PATH = "datasets/mdn_prepared"
REPO_URL = "https://github.com/mdn/content.git"

# Clone repo
if not os.path.exists(DATASET_DIR):
    os.makedirs("datasets", exist_ok=True)
    subprocess.run(["git", "clone", "--depth", "1", REPO_URL, DATASET_DIR], check=True)
else:
    print("MDN repo already exists, skipping clone.")

# Collect accessibility guide Markdown files
texts = []
for path in glob.glob(f"{DATASET_DIR}/files/en-us/web/accessibility/guides/**/*.md", recursive=True):
    try:
        with open(path, "r", encoding="utf-8") as f:
            content = f.read().strip()
            if content:
                texts.append({"text": content})
    except Exception as e:
        print(f"Skipping {path}: {e}")

# Save dataset
if texts:
    ds = Dataset.from_list(texts)
    ds.save_to_disk(OUTPUT_PATH)
    print(f"Saved {len(texts)} documents to {OUTPUT_PATH}")
else:
    print("No MDN markdown files found.")
