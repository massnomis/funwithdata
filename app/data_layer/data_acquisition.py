import os
import pandas as pd
import kagglehub
from pathlib import Path

CACHE_DIR = "data/cache"
RAW_DIR = "data/raw"

def kaggle_sync(dataset: str) -> str:
    """
    Sync dataset from Kaggle and cache locally.
    """
    cache_path = Path(CACHE_DIR)
    raw_path = Path(RAW_DIR)
    cache_path.mkdir(exist_ok=True, parents=True)
    raw_path.mkdir(exist_ok=True, parents=True)

    try:
        print("Downloading dataset...")
        path = kagglehub.dataset_download(dataset)
        return path
    except Exception as e:
        print(f"Error Downloading Kaggle Dataset: {e}")
        raise

def validate_cache(dataset_name: str, cache_file: str) -> pd.DataFrame:
    """
    Check for cached data to load.
    """
    try:
        file = os.path.join(CACHE_DIR, cache_file)
        if os.path.exists(file):
            print("Loading cached dataset...")
            return pd.read_pickle(file)
        return pd.DataFrame()
    except Exception:
        return pd.DataFrame()