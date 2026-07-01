"""PMC Scraper — Initial Build
============================
Scrapes a PubMed Central article and saves the text of each section
to a plain .txt file, one file per section.

You can also skip the scraper entirely and just paste text into a .txt file
manually — the output format is the same either way.

INSTALL:
pip install requests beautifulsoup4 lxml

RUN:
python scraper.py
"""

import re
import time
import random
import requests
from pathlib import Path
from bs4 import BeautifulSoup


# ─────────────────────────────────────────────────────────────────────────────
# HEADING LISTS
# Any heading that matches gets mapped to a canonical section name.
# Content between two headings belongs to the first heading.
# ─────────────────────────────────────────────────────────────────────────────

HEADING_MAP = {
    # abstract
    "abstract":                     "abstract",
    "summary":                      "abstract",
    "structured abstract":          "abstract",
    # introduction
    "introduction":                 "introduction",
    "background":                   "introduction",
    "background and introduction":  "introduction",
    # methods
    "methods":                      "methods",
    "materials and methods":        "methods",
    "online methods":               "methods",
    "methods and materials":        "methods",
    "patients and methods":         "methods",
    "subjects and methods":         "methods",
    "experimental procedures":      "methods",
    "methodology":                  "methods",
    "study design":                 "methods",
    # results
    "results":                      "results",
    "results and discussion":       "results",
    "findings":                     "results",
    # discussion
    "discussion":                   "discussion",
    "discussion and conclusions":   "discussion",
    "concluding remarks":           "discussion",
    # conclusion
    "conclusion":                   "conclusion",
    "conclusions":                  "conclusion",
}

# Headings to skip entirely — not useful content
SKIP = {
    "references", "bibliography", "supplementary", "supplemental",
    "author contributions", "acknowledgments", "acknowledgements",
    "conflict of interest", "competing interests", "data availability",
    "funding", "abbreviations", "ethics", "figure", "table",
    "declarations", "footnotes", "associated data", "supplementary information",
    "availability of data", "authors' contributions", "consent",
    "electronic supplementary", "publisher's note",
}


def classify_heading(raw_heading):
    """Map a heading string to a canonical section name.
    Returns None if the heading should be skipped."""
    text = raw_heading.strip().lower()

    # Skip list check first
    if any(skip in text for skip in SKIP):
        return None

    # Exact match
    if text in HEADING_MAP:
        return HEADING_MAP[text]

    # Starts-with match — e.g. "methods for rna-seq" → methods
    for heading, canonical in HEADING_MAP.items():
        if text.startswith(heading):
            return canonical

    # Keyword anywhere — e.g. "extended methods" → methods
    fallbacks = {
        "abstract": "abstract", "introduction": "introduction",
        "method":   "methods",  "material":     "methods",
        "result":   "results",  "finding":      "results",
        "discuss":  "discussion",
        "conclusion":  "conclusion",
    }
    for kw, canonical in fallbacks.items():
        if kw in text:
            return canonical

    # Unrecognized — keep it with the original heading as the name
    return raw_heading.strip().title()


# ─────────────────────────────────────────────────────────────────────────────
# TEXT CLEANING
# Remove citation numbers and extra whitespace before saving.
# ─────────────────────────────────────────────────────────────────────────────

def clean(text):
    text = re.sub(r"\s+",                                    " ",  text)
    text = re.sub(r"\[\d+(?:,\d+)*\]",                      "",   text)
    text = re.sub(r"\[\d+-\d+\]",                            "",   text)
    text = re.sub(r"\((Table|Figure|Fig\.?)\s+\w+(?:\s*\w)?\)", "",   text, flags=re.IGNORECASE)
    text = re.sub(r"\[\s*\]",                                "",   text)
    return text.strip()


# ─────────────────────────────────────────────────────────────────────────────
# FETCH
# Download the PMC webpage. Waits politely between requests.
# ─────────────────────────────────────────────────────────────────────────────

def fetch(pmcid, delay=1.5):
    """Download the HTML of a PMC article page."""
    pmcid = str(pmcid).upper().replace("PMC", "")
    url   = f"https://www.ncbi.nlm.nih.gov/pmc/articles/PMC{pmcid}/"

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        ),
        "Accept": "text/html,application/xhtml+xml",
    }

    wait = delay + random.random() * 1.0
    print(f"  Waiting {wait:.1f}s...")
    time.sleep(wait)

    print(f"  Fetching PMC{pmcid}...")
    try:
        r = requests.get(url, headers=headers, timeout=20)
        r.raise_for_status()
    except requests.exceptions.HTTPError as e:
        status = e.response.status_code if e.response else "?"
        hint   = ""
        if status == 403:
            hint = " (access denied — may not be open access)"
        elif status == 404:
            hint = " (not found — check the PMC ID)"
        raise RuntimeError(f"HTTP {status}{hint}") from e
    except requests.exceptions.Timeout:
        raise RuntimeError("Server timed out after 20s")

    return r.text


# ─────────────────────────────────────────────────────────────────────────────
# PARSE
# Extract sections from the HTML using heading lists.
# ─────────────────────────────────────────────────────────────────────────────

def parse(html):
    """Parse PMC HTML into a dict of {section_name: text}.

    Uses heading lists — content between heading N and heading N+1
    belongs to heading N."""
    soup = BeautifulSoup(html, "lxml")

    # Title
    title_tag = (
        soup.find("h1", class_="content-title") or
        soup.find("h1", class_="article-title") or
        soup.find("h1")
    )
    title = title_tag.get_text(strip=True) if title_tag else "Unknown Title"

    sections = {}

    # Abstract — can be <div class="abstract">, <section class="abstract">, etc.
    abstract_el = (
        soup.find("div", class_="abstract")       or
        soup.find("section", class_="abstract")   or
        soup.find("div", id="abstract")           or
        soup.find("div", class_="abstractSection")
    )
    if abstract_el:
        # Get only paragraph text from the abstract, not nested sections
        abstract_paras = abstract_el.find_all("p")
        stop_phrases = ["supplementary material", "keywords:", "the online version"]
        abstract_text_parts = []
        for p in abstract_paras:
            # Skip if inside a nested section
            parent_sec = p.find_parent("section")
            if parent_sec and parent_sec != abstract_el:
                continue
            p_text = p.get_text(" ", strip=True)
            if any(phrase in p_text.lower() for phrase in stop_phrases):
                break
            if p_text:
                abstract_text_parts.append(p_text)
        if abstract_text_parts:
            sections["abstract"] = clean(" ".join(abstract_text_parts))

    # Body sections — PMC uses either <div class="tsec"> or <section> elements
    section_divs = soup.find_all("div", class_="tsec")
    if not section_divs:
        # New PMC layout: find the main article body container first
        body_container = (
            soup.find("section", class_="body") or
            soup.find("section", class_="main-article-body") or
            soup
        )
        section_divs = [s for s in body_container.find_all("section")
                        if s.find(["h2", "h3", "h4"])]
    if not section_divs:
        section_divs = soup.find_all("div", id=re.compile(r"^[Ss]\d+"))

    # Track current top-level section for subsection merging
    current_top_section = None

    for div in section_divs:
        # Skip metadata sections (author info, citation data)
        aria_label = (div.get("aria-label") or "").lower()
        if "citation" in aria_label or "metadata" in aria_label:
            continue

        header = div.find(["h2", "h3", "h4"])
        if not header:
            continue

        heading_text = header.get_text(strip=True)
        canonical = classify_heading(heading_text)

        # Skip None (skip list) and abstract already captured above
        if canonical is None:
            continue
        if canonical == "abstract" and "abstract" in sections:
            continue

        # Determine if this is a top-level section or a subsection
        # h2 = top-level, h3/h4 = subsection
        is_top_level = header.name == "h2"

        if is_top_level:
            # Known canonical section — update tracker
            if canonical in ("abstract", "introduction", "methods", "results", "discussion", "conclusion"):
                current_top_section = canonical
            else:
                # Skip unrecognized top-level headings that look like author names
                # (contain commas typical of name lists)
                if "," in heading_text and len(heading_text.split(",")) >= 3:
                    continue
                current_top_section = canonical
        else:
            # Subsection: merge into current top-level section if unrecognized
            if canonical not in ("abstract", "introduction", "methods", "results", "discussion", "conclusion"):
                if current_top_section:
                    canonical = current_top_section
                # else keep the title-cased name

        text = clean(" ".join(
            p.get_text(" ", strip=True) for p in div.find_all("p")
        ))

        if not text or len(text) < 50:
            continue

        # Append — subsections merge into the same canonical section
        existing = sections.get(canonical, "")
        sections[canonical] = (existing + " " + text).strip() if existing else text

    return title, sections


# ─────────────────────────────────────────────────────────────────────────────
# SAVE
# Write each section to a .txt file in an output folder.
# ─────────────────────────────────────────────────────────────────────────────

def save(pmcid, title, sections, output_dir="scraped"):
    """Save each section as its own .txt file.

    Output structure:
        scraped/PMC12053221_title.txt
        PMC12053221_abstract.txt
        PMC12053221_introduction.txt
        PMC12053221_methods.txt
        PMC12053221_results.txt
        PMC12053221_discussion.txt
    """
    pmcid_clean = str(pmcid).upper().replace("PMC", "")
    folder      = Path(output_dir) / f"PMC{pmcid_clean}"
    folder.mkdir(parents=True, exist_ok=True)

    # Save title
    (folder / "title.txt").write_text(title, encoding="utf-8")

    # Save each section
    for section, text in sections.items():
        # Sanitize filename — remove characters invalid in file paths
        safe_name = re.sub(r'[/\\:*?"<>|]', '-', section)
        safe_name = safe_name.strip('. ')[:80]  # cap length too
        filename = f"{safe_name}.txt"
        (folder / filename).write_text(text, encoding="utf-8")
        print(f"  Saved: {folder}/{filename}  ({len(text.split())} words)")

    print(f"  All files saved to: {folder}/")
    return folder


# ─────────────────────────────────────────────────────────────────────────────
# SCRAPE — puts fetch + parse + save together
# ─────────────────────────────────────────────────────────────────────────────

def scrape(pmcid, output_dir="scraped"):
    """Scrape one PMC article and save each section as a .txt file.

    pmcid:      e.g. "PMC12053221" or "12053221"
    output_dir: folder where files are saved (created if it doesn't exist)
    """
    print(f"\nScraping PMC{pmcid}...")
    html            = fetch(pmcid)
    title, sections = parse(html)

    print(f"  Title: {title[:65]}{'...' if len(title) > 65 else ''}")
    print(f"  Sections found: {list(sections.keys())}")

    folder = save(pmcid, title, sections, output_dir)
    return folder


def scrape_multiple(pmcid_list, output_dir="scraped"):
    """Scrape a list of PMC articles one by one.
    Logs failures without stopping the whole batch."""
    success, failed = [], []

    for i, pmcid in enumerate(pmcid_list):
        print(f"\n[{i+1}/{len(pmcid_list)}]")
        try:
            folder = scrape(pmcid, output_dir)
            success.append(pmcid)
        except Exception as e:
            print(f"  FAILED: {e}")
            failed.append({"pmcid": pmcid, "error": str(e)})

    print(f"\nDone: {len(success)} succeeded, {len(failed)} failed")
    if failed:
        print("Failed:", [f["pmcid"] for f in failed])
    return success, failed


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # ── OPTION A: Scrape one paper ────────────────────────────────────────────
    scrape("PMC12053221")

    # ── OPTION B: Scrape multiple papers ─────────────────────────────────────
    # scrape_multiple([
    #     "PMC12053221",
    #     "PMC12359983",
    # ])

    # ── OPTION C: Skip the scraper, paste text manually ──────────────────────
    # Create a folder and drop your .txt files in directly:
    #   scraped/PMC12053221/abstract.txt
    #   scraped/PMC12053221/introduction.txt
    #   scraped/PMC12053221/results.txt
    #   scraped/PMC12053221/discussion.txt
    # The NLP step reads the same folder structure either way.
