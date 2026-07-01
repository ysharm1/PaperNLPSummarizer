#!/usr/bin/env bash
# Render build script — installs dependencies and downloads NLTK data

set -e

pip install -r requirements.txt
python -c "import nltk; nltk.download('punkt_tab'); nltk.download('stopwords')"
