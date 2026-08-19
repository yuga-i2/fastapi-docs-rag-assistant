"""
Bonus: query rewriting.

Ambiguous or under-specified questions ("how do I do auth?") often
retrieve poorly because the user's wording doesn't match the docs'
vocabulary. We ask the LLM to produce a couple of alternative phrasings
of the question, run retrieval for each, and fuse the results.

In demo mode (no LLM key) this is a documented no-op: it returns just the
original query, so the rest of the pipeline degrades gracefully.
"""

from __future__ import annotations

from app.llm import complete, _detect_provider

REWRITE_PROMPT = """The user asked a question about the FastAPI web
framework. Write 2 alternative ways to phrase this question that might
match different wording in technical documentation. Keep them short.
Return ONLY the alternatives, one per line, no numbering.

ORIGINAL QUESTION: {question}
""".strip()


def rewrite_query(question: str) -> list[str]:
    """Returns [original_question, *rewrites]. Rewrites are best-effort;
    any failure just falls back to the original question alone."""
    if _detect_provider() == "demo":
        return [question]
    try:
        raw = complete(REWRITE_PROMPT.format(question=question))
        alts = [line.strip("- ").strip() for line in raw.split("\n") if line.strip()]
        return [question] + alts[:2]
    except Exception:
        return [question]
