import pandas as pd
import os

def load_data(file_path):
    """
    Load the spam dataset from a given path.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset not found at {file_path}")
    
    # SMS Spam Collection is a tab-separated file with no header
    # Format: label \t text
    df = pd.read_csv(file_path, sep='\t', header=None, names=['label', 'text'])
    return df

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(current_dir, "..", "data", "raw", "SMSSpamCollection")
    
    try:
        df = load_data(data_path)
        print("Data loaded successfully.")
        print(f"Dataset shape: {df.shape}")
        print("Sample data:")
        print(df.head())
    except Exception as e:
        print(f"Error loading data: {e}")
