# Paper NLP Summarizer

Extractive NLP summarization pipeline for neurodegenerative research papers from PubMed Central.

Scrapes PMC articles, splits them into sections (abstract, introduction, methods, results, discussion), and produces extractive summaries using TextRank, LSA, LexRank, or Luhn.

## How It Works

```
PMC ID → scraper.py → scraped/PMC.../section.txt → summarizer.py → summaries
```

**Two aims:**
- **Aim 2** — summarize each section of each paper independently
- **Aim 1** — combine a section across multiple papers and summarize (e.g., "what do 10 papers say about GBA in their results?")

## Install

```bash
pip install requests beautifulsoup4 lxml sumy nltk
python -c "import nltk; nltk.download('punkt_tab'); nltk.download('stopwords')"
```

## Usage

### 1. Scrape a paper

```bash
python scraper.py
```

Edit the PMC ID at the bottom of `scraper.py`, or use it as a library:

```python
from scraper import scrape
scrape("PMC5934765")
```

Output goes to `scraped/PMC5934765/abstract.txt`, `results.txt`, etc.

### 2. Run summaries

```bash
python summarizer.py
```

This reads everything in `scraped/` and produces summaries. Output is printed and exported to `summaries.json`.

### 3. Or skip the scraper

Paste text directly into files:

```
scraped/PMC1234567/abstract.txt
scraped/PMC1234567/results.txt
scraped/PMC1234567/discussion.txt
```

The summarizer reads whatever `.txt` files are in the folder.

## Algorithms

| Algorithm | Best for |
|-----------|----------|
| TextRank | Results, Discussion (best all-around) |
| LSA | Long Methods sections (topic diversity) |
| LexRank | Short Abstracts, Introductions |
| Luhn | Quick triage of many papers |

## Demo Output

See `scraped/` for example papers and `summaries.json` for the generated summaries.

**PMC12053221** — Oligodendroglia vulnerability in Parkinson's disease (sn-RNA-seq)  
**PMC5934765** — Alpha-synuclein aggregates activate calcium pump SERCA

## Project Context

Built for [PARK-seq](https://parkseq.com) — a curated database of genes implicated in Parkinson's disease from sequencing/omics/array studies. This pipeline automates literature summarization across the papers in that database.
