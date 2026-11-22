import os
import urllib.request
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
DATASET_PATH = os.path.join(DATA_DIR, "dataset.csv")

PRIMARY_URL = "https://raw.githubusercontent.com/codebasics/py/master/ML/14_naive_bayes/spam.csv"
FALLBACK_URL = "https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv"

def download_data():
    """Download dataset if not already present."""
    os.makedirs(DATA_DIR, exist_ok=True)
    if os.path.exists(DATASET_PATH) and os.path.getsize(DATASET_PATH) > 1000:
        return DATASET_PATH
    
    print(f"Dataset not found locally. Downloading from primary source: {PRIMARY_URL}")
    try:
        urllib.request.urlretrieve(PRIMARY_URL, DATASET_PATH)
        print("Dataset downloaded successfully.")
        return DATASET_PATH
    except Exception as e:
        print(f"Primary download failed ({e}). Attempting fallback URL...")
        try:
            temp_tsv = os.path.join(DATA_DIR, "sms.tsv")
            urllib.request.urlretrieve(FALLBACK_URL, temp_tsv)
            df = pd.read_csv(temp_tsv, sep='\t', header=None, names=['label', 'text'])
            df.to_csv(DATASET_PATH, index=False)
            if os.path.exists(temp_tsv):
                os.remove(temp_tsv)
            print("Fallback dataset downloaded and converted.")
            return DATASET_PATH
        except Exception as e2:
            raise RuntimeError(f"Failed to acquire dataset from all sources: {e2}")

def load_data(file_path: str = None) -> pd.DataFrame:
    """Load and normalize spam dataset."""
    path = file_path or download_data()
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset file missing at {path}")
    
    # Check if CSV or TSV
    try:
        df = pd.read_csv(path)
        if 'Category' in df.columns and 'Message' in df.columns:
            df = df.rename(columns={'Category': 'label', 'Message': 'text'})
    except Exception:
        df = pd.read_csv(path, sep='\t', header=None, names=['label', 'text'])
        
    df['label'] = df['label'].astype(str).str.strip().str.lower()
    df['label'] = df['label'].map({'ham': 0, 'spam': 1})
    df = df.dropna(subset=['label', 'text'])
    df['label'] = df['label'].astype(int)
    return df

if __name__ == "__main__":
    df = load_data()
    print("Data loaded:", df.shape)
    print("Label distribution:\n", df['label'].value_counts())
