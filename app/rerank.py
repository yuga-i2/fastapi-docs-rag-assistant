"""
Bonus: document re-ranking.

Hybrid search (search.py) is optimized for recall across a large corpus
cheaply. Reranking takes a slightly larger candidate set from hybrid
search and re-scores each candidate's relevance to the *exact* question
using the LLM, which is slower but more precise -- a common two-stage
retrieval pattern (retrieve broad, then rerank narrow).

In demo mode (no LLM key) this degrades gracefully to a documented
no-op: it returns the hybrid-search order unchanged.
"""

from __future__ import annotations

import re

from app.llm import complete, _detect_provider

RERANK_PROMPT = """Question: {question}

Passage ({source} | {heading}):
{text}

On a scale of 0-10, how directly does this passage answer the question?
Respond with ONLY the number.
""".strip()


def rerank(question: str, candidates: list[dict], top_k: int = 5) -> list[dict]:
    if _detect_provider() == "demo" or not candidates:
        return candidates[:top_k]

    scored = []
    for c in candidates:
        try:
            raw = complete(RERANK_PROMPT.format(
                question=question, source=c["source"], heading=c["heading"],
                text=c["text"][:600],
            ))
            match = re.search(r"\d+(\.\d+)?", raw)
            score = float(match.group(0)) if match else 0.0
        except Exception:
            score = 0.0
        scored.append({**c, "rerank_score": score})

    scored.sort(key=lambda c: c["rerank_score"], reverse=True)
    return scored[:top_k]
