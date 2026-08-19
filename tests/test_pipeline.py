"""
Minimal smoke tests for the pipeline. Run with: pytest tests/

These don't require an LLM API key -- they run against the demo-mode
LLM fallback, so they can run in CI with zero secrets configured.
"""

import json
from pathlib import Path

from app.rag import answer_question
from app.search import get_index

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def test_chunks_exist():
    chunks_file = DATA_DIR / "chunks.json"
    assert chunks_file.exists(), "run `python -m app.ingest` first"
    chunks = json.loads(chunks_file.read_text())
    assert len(chunks) > 100
    assert all("text" in c and "source" in c for c in chunks)


def test_hybrid_search_returns_relevant_result():
    index = get_index()
    results = index.search("how do background tasks work", top_k=5, method="hybrid")
    assert len(results) == 5
    sources = [r["source"] for r in results]
    assert "tutorial/background-tasks.md" in sources


def test_keyword_and_vector_search_run():
    index = get_index()
    kw = index.search("docker deployment", top_k=3, method="keyword")
    vec = index.search("docker deployment", top_k=3, method="vector")
    assert len(kw) == 3
    assert len(vec) == 3


def test_rag_pipeline_end_to_end():
    result = answer_question("How do I add a path parameter?")
    assert result["answer"]
    assert len(result["sources"]) > 0
    assert result["retrieval_method"] == "hybrid"
