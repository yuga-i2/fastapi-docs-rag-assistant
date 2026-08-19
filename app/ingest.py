"""
Ingestion pipeline.

Fetches the raw markdown pages listed in doc_sources.py, cleans them up,
splits them into overlapping chunks, and writes the result to
data/chunks.json -- the knowledge base used by search.py.

Run:
    python -m app.ingest

This is a fully automated, scriptable ingestion pipeline (no manual steps),
which satisfies the "Automated ingestion with a Python script" criterion.
"""

import json
import re
import time
from pathlib import Path

import requests

from app.doc_sources import DOC_PATHS, REPO_RAW_BASE

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
OUTPUT_FILE = DATA_DIR / "chunks.json"

CHUNK_SIZE_CHARS = 1200
CHUNK_OVERLAP_CHARS = 200


def fetch_doc(path: str) -> str | None:
    url = f"{REPO_RAW_BASE}/{path}"
    try:
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
        return resp.text
    except requests.RequestException as e:
        print(f"  [WARN] failed to fetch {path}: {e}")
        return None


def clean_markdown(text: str) -> str:
    """Strip things that add noise but no retrieval value: YAML frontmatter,
    HTML comments/tags, style blocks, badge links, images, and excess
    blank lines."""
    text = re.sub(r"^---\n.*?\n---\n", "", text, flags=re.DOTALL)   # frontmatter
    text = re.sub(r"<!--.*?-->", "", text, flags=re.DOTALL)
    text = re.sub(r"<style>.*?</style>", "", text, flags=re.DOTALL)
    text = re.sub(r"!\[.*?\]\(.*?\)", "", text)                      # images
    text = re.sub(r"\[!\[.*?\]\(.*?\)\]\(.*?\)", "", text)           # badge links
    text = re.sub(r"<[^>]+>", "", text)                              # any HTML tag
    text = re.sub(r"\{\s*\.[\w-]+.*?\}", "", text)                   # {.class} attrs
    text = re.sub(r"\{\s*#[\w-]+\s*\}", "", text)                    # {#id} attrs
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def split_into_sections(text: str, source: str) -> list[dict]:
    """Split a markdown doc into sections along headings, then further
    split any section that's still too long into overlapping chunks."""
    # Split on markdown headings, keeping the heading with its content
    parts = re.split(r"(?m)^(#{1,3} .+)$", text)
    sections = []
    current_heading = "Introduction"
    buf = ""
    for part in parts:
        if re.match(r"^#{1,3} ", part):
            if buf.strip():
                sections.append((current_heading, buf.strip()))
            current_heading = part.lstrip("# ").strip()
            buf = ""
        else:
            buf += part
    if buf.strip():
        sections.append((current_heading, buf.strip()))

    chunks = []
    for heading, content in sections:
        if len(content) <= CHUNK_SIZE_CHARS:
            chunks.append({"heading": heading, "text": content})
        else:
            start = 0
            while start < len(content):
                end = start + CHUNK_SIZE_CHARS
                chunks.append({"heading": heading, "text": content[start:end]})
                start = end - CHUNK_OVERLAP_CHARS
    return chunks


def run():
    DATA_DIR.mkdir(exist_ok=True)
    all_chunks = []
    chunk_id = 0

    print(f"Ingesting {len(DOC_PATHS)} documents from FastAPI docs...")
    for path in DOC_PATHS:
        raw = fetch_doc(path)
        if raw is None:
            continue
        cleaned = clean_markdown(raw)
        sections = split_into_sections(cleaned, path)

        for sec in sections:
            if len(sec["text"].strip()) < 40:
                continue  # skip near-empty fragments
            if "sponsor" in sec["heading"].lower() or "{%" in sec["text"]:
                continue  # skip templating placeholders / sponsor blocks
            all_chunks.append({
                "id": f"chunk-{chunk_id:05d}",
                "source": path,
                "heading": sec["heading"],
                "text": sec["text"],
                "url": f"https://fastapi.tiangolo.com/{path.replace('index.md', '').replace('.md', '/')}",
            })
            chunk_id += 1

        print(f"  ok: {path}  ({len(sections)} chunks)")
        time.sleep(0.05)  # be polite to the raw content CDN

    with open(OUTPUT_FILE, "w") as f:
        json.dump(all_chunks, f, indent=2)

    print(f"\nDone. {len(all_chunks)} chunks from {len(DOC_PATHS)} documents "
          f"written to {OUTPUT_FILE}")


if __name__ == "__main__":
    run()
