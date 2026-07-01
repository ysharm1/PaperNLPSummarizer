#!/usr/bin/env bash
# Render build script — installs dependencies and downloads NLTK data

set -e

pip install -r requirements.txt

# Download NLTK data to a known location within the project
python -c "
import nltk
import os
nltk_dir = os.path.join(os.getcwd(), 'nltk_data')
os.makedirs(nltk_dir, exist_ok=True)
nltk.download('punkt_tab', download_dir=nltk_dir)
nltk.download('stopwords', download_dir=nltk_dir)
print(f'NLTK data downloaded to: {nltk_dir}')
"
