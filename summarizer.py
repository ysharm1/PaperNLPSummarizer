"""NLP Summarizer — reads scraped .txt files and produces extractive summaries.
==============================================================================

Reads from the scraped/ folder (output of scraper.py) and runs extractive
summarization on each section using sumy (TextRank, LSA, LexRank, Luhn).

Supports two aims:
  Aim 1 — combine one section across multiple papers, summarize together
  Aim 2 — summarize each section of each paper independently

INSTALL:
pip install sumy nltk
python -c "import nltk; nltk.download('punkt_tab'); nltk.download('stopwords')"

RUN:
python summarizer.py
"""

import os
import re
from pathlib import Path

import nltk

# Add local nltk_data if it exists (for Render deployment)
_nltk_local = os.path.join(os.path.dirname(os.path.abspath(__file__)), "nltk_data")
if os.path.exists(_nltk_local):
    nltk.data.path.insert(0, _nltk_local)

from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.text_rank import TextRankSummarizer
from sumy.summarizers.lsa import LsaSummarizer
from sumy.summarizers.lex_rank import LexRankSummarizer
from sumy.summarizers.luhn import LuhnSummarizer


# ─────────────────────────────────────────────────────────────────────────────
# ALGORITHMS
# ─────────────────────────────────────────────────────────────────────────────

ALGORITHMS = {
    "textrank": TextRankSummarizer,
    "lsa":      LsaSummarizer,
    "lexrank":  LexRankSummarizer,
    "luhn":     LuhnSummarizer,
}


# ─────────────────────────────────────────────────────────────────────────────
# CORE: summarize a block of text
# ─────────────────────────────────────────────────────────────────────────────

def clean_for_nlp(text):
    """Remove figure/table legends and other noise before summarization."""
    lines = text.split(". ")
    cleaned = []
    for line in lines:
        # Skip figure/table legend sentences
        if re.match(r"^(Fig(ure)?|Table|Heatmap|Bar chart|Violin plot|UMAP|Plot|Schematic)\s", line, re.IGNORECASE):
            continue
        # Skip lines that are mostly about asterisks, scale bars, etc.
        if re.match(r"^(Asterisks?|Scale bar|Arrow|Dotted line|Data are)", line, re.IGNORECASE):
            continue
        cleaned.append(line)
    return ". ".join(cleaned)


def summarize(text, n_sentences=5, method="textrank"):
    """Extractive summarization — returns the N most important sentences.

    text:        any string (section content, combined sections, etc.)
    n_sentences: how many sentences to return
    method:      textrank | lsa | lexrank | luhn

    Returns a list of unique sentence strings in original document order.
    """
    if not text or not text.strip():
        return []

    # Clean out figure legends before summarizing
    text = clean_for_nlp(text)

    if not text.strip():
        return []

    # Cap to actual sentence count so sumy doesn't error on short text
    rough_count = len(re.findall(r"[.!?]+\s", text)) + 1
    n_sentences = min(n_sentences, max(1, rough_count))

    # Request extra sentences to handle deduplication
    request_n = min(n_sentences + 3, rough_count)

    parser = PlaintextParser.from_string(text, Tokenizer("english"))
    summarizer_class = ALGORITHMS.get(method, TextRankSummarizer)
    summarizer = summarizer_class()

    try:
        raw_sentences = summarizer(parser.document, request_n)
        # Deduplicate while preserving order
        seen = set()
        unique = []
        for s in raw_sentences:
            s_str = str(s).strip()
            if s_str not in seen and len(s_str) > 20:
                seen.add(s_str)
                unique.append(s_str)
            if len(unique) >= n_sentences:
                break
        return unique
    except Exception as e:
        return [f"[Summarization error: {e}]"]


# ─────────────────────────────────────────────────────────────────────────────
# LOAD: read .txt files from scraped/ folder
# ─────────────────────────────────────────────────────────────────────────────

def load_paper(paper_dir):
    """Load all section .txt files from a scraped paper directory.

    Returns dict: {"abstract": "text...", "results": "text...", ...}
    """
    paper_dir = Path(paper_dir)
    sections = {}
    for txt_file in sorted(paper_dir.glob("*.txt")):
        section_name = txt_file.stem  # filename without .txt
        if section_name == "title":
            continue  # skip title file
        text = txt_file.read_text(encoding="utf-8").strip()
        if text:
            sections[section_name] = text
    return sections


def load_all_papers(scraped_dir="scraped"):
    """Load all papers from the scraped directory.

    Returns dict: {"PMC12053221": {"abstract": "...", "results": "..."}, ...}
    """
    scraped_dir = Path(scraped_dir)
    papers = {}
    for paper_dir in sorted(scraped_dir.iterdir()):
        if paper_dir.is_dir():
            pmcid = paper_dir.name  # e.g. "PMC12053221"
            sections = load_paper(paper_dir)
            if sections:
                papers[pmcid] = sections
    return papers


def get_title(paper_dir):
    """Read the title from a paper directory."""
    title_file = Path(paper_dir) / "title.txt"
    if title_file.exists():
        return title_file.read_text(encoding="utf-8").strip()
    return "Unknown Title"


# ─────────────────────────────────────────────────────────────────────────────
# AIM 2: summarize each section of each paper
# ─────────────────────────────────────────────────────────────────────────────

def summarize_paper(sections, n_sentences=5, method="textrank",
                    target_sections=("abstract", "introduction", "results", "discussion")):
    """Summarize specific sections of a single paper.

    sections:        dict of {section_name: text}
    target_sections: which sections to summarize (others skipped)
    n_sentences:     sentences per section

    Returns dict: {"abstract": [sentences], "results": [sentences], ...}
    """
    results = {}
    for section in target_sections:
        text = sections.get(section, "")
        if text and len(text.split()) > 20:  # skip very short sections
            results[section] = summarize(text, n_sentences, method)
        else:
            results[section] = [f"[{section} not found or too short]"]
    return results


# ─────────────────────────────────────────────────────────────────────────────
# AIM 1: combine one section across papers, summarize together
# ─────────────────────────────────────────────────────────────────────────────

def summarize_across_papers(papers, section="results", n_sentences=5, method="textrank"):
    """Combine a section from all papers and summarize the combined text.

    papers:      dict from load_all_papers()
    section:     which section to combine (e.g. "results")
    n_sentences: how many sentences from the combined text

    Returns list of sentences (the global summary).
    """
    combined = []
    for pmcid, sections in papers.items():
        text = sections.get(section, "")
        if text:
            combined.append(text)

    if not combined:
        return [f"[No papers have a '{section}' section]"]

    combined_text = " ".join(combined)
    return summarize(combined_text, n_sentences, method)


# ─────────────────────────────────────────────────────────────────────────────
# PRINT HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def print_paper_summary(pmcid, summary_result, title=""):
    """Pretty-print a single paper's section summaries."""
    print(f"\n{'═' * 70}")
    print(f"  {pmcid}")
    if title:
        truncated = title[:66] + ("..." if len(title) > 66 else "")
        print(f"  {truncated}")
    print(f"{'═' * 70}")

    for section, sentences in summary_result.items():
        print(f"\n  ── {section.upper()} ({len(sentences)} sentences) ──")
        for i, s in enumerate(sentences, 1):
            # Wrap long sentences
            if len(s) > 100:
                print(f"  {i}. {s[:97]}...")
            else:
                print(f"  {i}. {s}")


def print_global_summary(section, sentences, paper_count):
    """Pretty-print a cross-paper summary."""
    print(f"\n{'═' * 70}")
    print(f"  GLOBAL SUMMARY: {section.upper()} across {paper_count} papers")
    print(f"{'═' * 70}")
    for i, s in enumerate(sentences, 1):
        if len(s) > 100:
            print(f"  {i}. {s[:97]}...")
        else:
            print(f"  {i}. {s}")


# ─────────────────────────────────────────────────────────────────────────────
# EXPORT
# ─────────────────────────────────────────────────────────────────────────────

def export_json(results, output_file="summaries.json"):
    """Export summary results to JSON."""
    import json
    Path(output_file).write_text(
        json.dumps(results, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )
    print(f"\nExported to {output_file}")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    scraped_dir = Path("scraped")

    if not scraped_dir.exists() or not list(scraped_dir.iterdir()):
        print("No scraped papers found in scraped/ directory.")
        print("Run scraper.py first, or paste .txt files into scraped/PMC<ID>/")
        exit(1)

    # Load all papers
    papers = load_all_papers(scraped_dir)
    print(f"Loaded {len(papers)} paper(s) from {scraped_dir}/")
    for pmcid, sections in papers.items():
        print(f"  {pmcid}: {list(sections.keys())}")

    # ── AIM 2: Per-paper summaries ───────────────────────────────────────────
    print(f"\n{'─' * 70}")
    print("AIM 2: Per-paper section summaries (TextRank, 3 sentences each)")
    print("─" * 70)

    all_summaries = {}
    for pmcid, sections in papers.items():
        title = get_title(scraped_dir / pmcid)
        result = summarize_paper(sections, n_sentences=3, method="textrank")
        all_summaries[pmcid] = result
        print_paper_summary(pmcid, result, title=title)

    # ── AIM 1: Cross-paper summary ──────────────────────────────────────────
    if len(papers) > 1:
        print(f"\n{'─' * 70}")
        print("AIM 1: Cross-paper summary (Results section, TextRank, 5 sentences)")
        print("─" * 70)

        global_summary = summarize_across_papers(
            papers, section="results", n_sentences=5, method="textrank"
        )
        print_global_summary("results", global_summary, len(papers))
    else:
        print(f"\n(Only 1 paper loaded — Aim 1 cross-paper summary needs 2+ papers)")

    # ── Method comparison ────────────────────────────────────────────────────
    print(f"\n{'─' * 70}")
    print("METHOD COMPARISON: Results section, 2 sentences each")
    print("─" * 70)

    # Use the first paper's results
    first_pmcid = list(papers.keys())[0]
    results_text = papers[first_pmcid].get("results", "")
    if results_text:
        for method in ["textrank", "lsa", "lexrank", "luhn"]:
            sentences = summarize(results_text, n_sentences=2, method=method)
            print(f"\n  [{method.upper()}]")
            for s in sentences:
                print(f"    • {s[:90]}...")
    else:
        print("  (No results section found)")

    # ── Export ───────────────────────────────────────────────────────────────
    export_json(all_summaries)
