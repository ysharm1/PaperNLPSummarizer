"""Flask app — simple web frontend for the NLP summarizer.

Run:
    python app.py

Then open http://localhost:5000 in your browser.
"""

import json
from pathlib import Path
from flask import Flask, request, jsonify, send_from_directory

from scraper import scrape
from summarizer import (
    load_paper, load_all_papers, summarize_paper,
    summarize_across_papers, get_title, export_json
)

app = Flask(__name__, static_folder="static")


@app.route("/")
def index():
    return send_from_directory("static", "index.html")


@app.route("/api/papers", methods=["GET"])
def list_papers():
    """List all scraped papers."""
    scraped_dir = Path("scraped")
    papers = []
    if scraped_dir.exists():
        for d in sorted(scraped_dir.iterdir()):
            if d.is_dir():
                title = get_title(d)
                sections = [f.stem for f in d.glob("*.txt") if f.stem != "title"]
                papers.append({
                    "pmcid": d.name,
                    "title": title,
                    "sections": sections,
                    "word_count": sum(
                        len((d / f"{s}.txt").read_text().split())
                        for s in sections if (d / f"{s}.txt").exists()
                    )
                })
    return jsonify(papers)


@app.route("/api/scrape", methods=["POST"])
def scrape_paper():
    """Scrape a PMC paper by ID."""
    data = request.get_json()
    pmcid = data.get("pmcid", "").strip()

    if not pmcid:
        return jsonify({"error": "No PMC ID provided"}), 400

    # Check if already scraped
    paper_dir = Path("scraped") / pmcid.upper().replace("PMC", "PMC")
    if not pmcid.upper().startswith("PMC"):
        pmcid = f"PMC{pmcid}"

    try:
        folder = scrape(pmcid)
        title = get_title(folder)
        sections = [f.stem for f in folder.glob("*.txt") if f.stem != "title"]
        return jsonify({
            "pmcid": folder.name,
            "title": title,
            "sections": sections,
            "message": f"Scraped successfully: {len(sections)} sections"
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/summarize", methods=["POST"])
def summarize():
    """Summarize papers."""
    data = request.get_json()
    pmcids = data.get("pmcids", [])
    method = data.get("method", "textrank")
    n_sentences = data.get("n_sentences", 3)
    aim = data.get("aim", 2)
    section = data.get("section", "results")  # for aim 1

    if not pmcids:
        return jsonify({"error": "No papers selected"}), 400

    results = {}

    if aim == 2:
        # Per-paper summaries
        for pmcid in pmcids:
            paper_dir = Path("scraped") / pmcid
            if not paper_dir.exists():
                results[pmcid] = {"error": f"Not scraped yet"}
                continue
            sections = load_paper(paper_dir)
            title = get_title(paper_dir)
            summary = summarize_paper(sections, n_sentences=n_sentences, method=method)
            results[pmcid] = {
                "title": title,
                "summaries": summary
            }
    else:
        # Cross-paper summary (Aim 1)
        papers = {}
        for pmcid in pmcids:
            paper_dir = Path("scraped") / pmcid
            if paper_dir.exists():
                papers[pmcid] = load_paper(paper_dir)

        if not papers:
            return jsonify({"error": "No scraped papers found"}), 400

        sentences = summarize_across_papers(
            papers, section=section, n_sentences=n_sentences, method=method
        )
        results = {
            "aim": 1,
            "section": section,
            "papers_used": list(papers.keys()),
            "summary": sentences
        }

    return jsonify(results)


if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("FLASK_ENV") != "production"
    print("Starting Paper NLP Summarizer...")
    print(f"Open http://localhost:{port} in your browser")
    app.run(debug=debug, host="0.0.0.0", port=port)
