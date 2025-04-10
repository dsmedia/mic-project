#!/usr/bin/env python3
"""
Script to download the corpus of news articles for MIC analysis.
This script fetches the zip file containing news articles and extracts it to the data/raw directory.
"""

import os
import urllib.request
import zipfile
import logging
import requests
from tqdm import tqdm
from pathlib import Path

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Define file paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
ZIP_PATH = RAW_DATA_DIR / "news_articles.zip"

# URL for the corpus
CORPUS_URL = "https://www.dropbox.com/scl/fo/6dtw8wafbengbze4am7ft/AHUl4WVv-619PJ2YwVFFd1k?rlkey=puwzr74w10ac3lsyom0pfd4y5&st=gydjujqv&dl=1"


def download_file(url, destination):
    """Download a file from URL to destination path with progress bar."""
    logger.info(f"Downloading {url} to {destination}")
    try:
        # Create the parent directory if it doesn't exist
        os.makedirs(os.path.dirname(destination), exist_ok=True)
        
        # Download with progress bar
        response = requests.get(url, stream=True)
        total_size = int(response.headers.get('content-length', 0))
        block_size = 1024  # 1 Kibibyte
        
        with open(destination, 'wb') as f:
            progress_bar = tqdm(total=total_size, unit='iB', unit_scale=True)
            for data in response.iter_content(block_size):
                progress_bar.update(len(data))
                f.write(data)
            progress_bar.close()
            
        logger.info(f"Successfully downloaded to {destination}")
        return True
    except Exception as e:
        logger.error(f"Failed to download {url}: {e}")
        return False


def extract_zip(zip_path, extract_to):
    """Extract a zip file to specified directory with progress bar."""
    logger.info(f"Extracting {zip_path} to {extract_to}")
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            # Get list of files for progress tracking
            file_list = zip_ref.namelist()
            total_files = len(file_list)
            
            # Create progress bar
            with tqdm(total=total_files, desc="Extracting files") as progress_bar:
                for file in file_list:
                    zip_ref.extract(file, extract_to)
                    progress_bar.update(1)
                    
        logger.info(f"Successfully extracted to {extract_to}")
        return True
    except Exception as e:
        logger.error(f"Failed to extract {zip_path}: {e}")
        return False


def main():
    """Main function to download and extract the news articles corpus."""
    logger.info("Starting download of news articles corpus")
    
    # Check if zip file already exists
    if ZIP_PATH.exists():
        logger.info(f"Zip file already exists at {ZIP_PATH}. Skipping download.")
    else:
        # Download the zip file
        logger.info(f"Zip file not found. Downloading from {CORPUS_URL}")
        success = download_file(CORPUS_URL, ZIP_PATH)
        if not success:
            logger.error("Failed to download the corpus. Exiting.")
            return
    
    # Extract the zip file
    success = extract_zip(ZIP_PATH, RAW_DATA_DIR)
    if not success:
        logger.error("Failed to extract the corpus. Exiting.")
        return
    
    logger.info("Successfully downloaded and extracted the news articles corpus")


if __name__ == "__main__":
    main()