import urllib.request
from pathlib import Path

URL      = "https://dataverse.harvard.edu/api/access/datafile/5194114"
SAVE_PATH = Path("data/5194114.npz")

def download():
    if SAVE_PATH.exists():
        print("Data already exists, skipping download.")
        return
    print("Downloading dataset...")
    urllib.request.urlretrieve(URL, SAVE_PATH)
    print(f"Saved to {SAVE_PATH}")

if __name__ == "__main__":
    download()
