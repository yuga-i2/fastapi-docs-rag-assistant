"""
Hybrid retrieval engine: keyword search (BM25) + vector search, combined
via Reciprocal Rank Fusion (RRF).

Design notes
------------
- Keyword search uses BM25Okapi (rank_bm25), a strong, dependency-light
  keyword-search baseline.
- Vector search uses TF-IDF + Truncated SVD (a "poor man's embedding"),
  built entirely locally with scikit-learn. This means the whole project
  runs with NO external API calls and NO large model downloads -- it's
  fully reproducible offline.
  -> To upgrade retrieval quality, swap `_build_vectors` to call a real
     embedding model (OpenAI text-embedding-3-small, Cohere, or a local
     sentence-transformers model) -- the rest of the pipeline (indexing,
     RRF fusion, evaluation) does not need to change. This tradeoff is
     documented in the README.
- Combining both retrieval signals is what the course calls "hybrid
  search": each ranks candidates differently (lexical overlap vs.
  semantic similarity), and fusing them is more robust than either alone.
- An optional LLM-based query rewriting step (rewrite.py) can be layered
  on top of this search for out-of-vocabulary or ambiguous questions.
"""

from __future__ import annotations

import json
import pickle
import re
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
CHUNKS_FILE = DATA_DIR / "chunks.json"
INDEX_FILE = DATA_DIR / "search_index.pkl"

TOKEN_RE = re.compile(r"[a-z0-9_]+")


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


class HybridSearch:
    def __init__(self, chunks: list[dict]):
        self.chunks = chunks
        self.texts = [f"{c['heading']} {c['text']}" for c in chunks]

        # --- keyword index (BM25) ---
        tokenized = [tokenize(t) for t in self.texts]
        self.bm25 = BM25Okapi(tokenized)

        # --- vector index (TF-IDF -> SVD, cosine similarity) ---
        self.vectorizer = TfidfVectorizer(max_features=10_000, ngram_range=(1, 2))
        tfidf = self.vectorizer.fit_transform(self.texts)
        n_components = min(150, tfidf.shape[0] - 1, tfidf.shape[1] - 1)
        self.svd = TruncatedSVD(n_components=n_components, random_state=42)
        self.doc_vectors = self.svd.fit_transform(tfidf)

    # -- individual retrieval strategies -----------------------------------

    def keyword_search(self, query: str, top_k: int = 10) -> list[tuple[int, float]]:
        scores = self.bm25.get_scores(tokenize(query))
        ranked = np.argsort(scores)[::-1][:top_k]
        return [(int(i), float(scores[i])) for i in ranked if scores[i] > 0]

    def vector_search(self, query: str, top_k: int = 10) -> list[tuple[int, float]]:
        q_tfidf = self.vectorizer.transform([query])
        q_vec = self.svd.transform(q_tfidf)
        sims = cosine_similarity(q_vec, self.doc_vectors)[0]
        ranked = np.argsort(sims)[::-1][:top_k]
        return [(int(i), float(sims[i])) for i in ranked]

    # -- hybrid fusion --------------------------------------------------

    def hybrid_search(self, query: str, top_k: int = 5, k_rrf: int = 60) -> list[dict]:
        """Reciprocal Rank Fusion of keyword + vector rankings."""
        kw = self.keyword_search(query, top_k=30)
        vec = self.vector_search(query, top_k=30)

        rrf_scores: dict[int, float] = {}
        for rank, (idx, _) in enumerate(kw):
            rrf_scores[idx] = rrf_scores.get(idx, 0.0) + 1.0 / (k_rrf + rank + 1)
        for rank, (idx, _) in enumerate(vec):
            rrf_scores[idx] = rrf_scores.get(idx, 0.0) + 1.0 / (k_rrf + rank + 1)

        ranked = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        results = []
        for idx, score in ranked:
            chunk = dict(self.chunks[idx])
            chunk["score"] = score
            results.append(chunk)
        return results

    def search(self, query: str, top_k: int = 5, method: str = "hybrid") -> list[dict]:
        if method == "keyword":
            hits = self.keyword_search(query, top_k)
        elif method == "vector":
            hits = self.vector_search(query, top_k)
        else:
            return self.hybrid_search(query, top_k)
        return [{**self.chunks[i], "score": s} for i, s in hits]

    # -- persistence ------------------------------------------------------

    def save(self, path: Path = INDEX_FILE):
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path: Path = INDEX_FILE) -> "HybridSearch":
        with open(path, "rb") as f:
            return pickle.load(f)


def build_index() -> HybridSearch:
    with open(CHUNKS_FILE) as f:
        chunks = json.load(f)
    index = HybridSearch(chunks)
    index.save()
    print(f"Built hybrid search index over {len(chunks)} chunks -> {INDEX_FILE}")
    return index


def get_index() -> HybridSearch:
    if INDEX_FILE.exists():
        return HybridSearch.load()
    return build_index()


if __name__ == "__main__":
    idx = build_index()
    for r in idx.search("how do I add a background task", method="hybrid"):
        print(f"{r['score']:.4f}  {r['source']}  |  {r['heading']}")
