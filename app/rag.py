"""
Core RAG flow: retrieve context -> build prompt -> call LLM -> return
a grounded answer with citations back to source docs.
"""

from __future__ import annotations

from app.llm import complete
from app.rerank import rerank as rerank_candidates
from app.rewrite import rewrite_query
from app.search import HybridSearch, get_index

SYSTEM_PROMPT = """You are a helpful assistant that answers questions about
the FastAPI web framework, using ONLY the documentation excerpts provided
in the CONTEXT. If the CONTEXT does not contain enough information to
answer, say so clearly instead of guessing. Keep answers concise and
include short code examples when the docs provide them. Reference the
source file(s) you used."""

PROMPT_TEMPLATE = """Answer the QUESTION based on the CONTEXT from the
FastAPI documentation. Only use facts from the CONTEXT.

QUESTION: {question}

CONTEXT:
{context}
""".strip()


def build_context(chunks: list[dict]) -> str:
    parts = []
    for c in chunks:
        parts.append(f"[source: {c['source']} | {c['heading']}]\n{c['text']}")
    return "\n\n".join(parts)


def answer_question(
    question: str,
    index: HybridSearch | None = None,
    top_k: int = 5,
    method: str = "hybrid",
    use_query_rewriting: bool = False,
    use_reranking: bool = False,
) -> dict:
    """Full RAG call. Returns the answer plus the retrieved chunks used,
    so the UI can show citations and monitoring can log what was retrieved.

    use_query_rewriting: expand the question into a few alternative
        phrasings and fuse retrieval across all of them (bonus feature,
        see rewrite.py). No-ops gracefully in demo mode.
    use_reranking: retrieve a larger candidate set with hybrid search,
        then have the LLM re-score and reorder them before truncating to
        top_k (bonus feature, see rerank.py). No-ops gracefully in demo mode.
    """
    index = index or get_index()

    if use_query_rewriting:
        queries = rewrite_query(question)
        fetch_k = (top_k * 3) if use_reranking else top_k
        seen: dict[str, dict] = {}
        for q in queries:
            for c in index.search(q, top_k=fetch_k, method=method):
                if c["id"] not in seen or c["score"] > seen[c["id"]]["score"]:
                    seen[c["id"]] = c
        chunks = sorted(seen.values(), key=lambda c: c["score"], reverse=True)
    else:
        fetch_k = (top_k * 3) if use_reranking else top_k
        chunks = index.search(question, top_k=fetch_k, method=method)

    if use_reranking:
        chunks = rerank_candidates(question, chunks, top_k=top_k)
    else:
        chunks = chunks[:top_k]

    context = build_context(chunks)
    prompt = PROMPT_TEMPLATE.format(question=question, context=context)
    answer = complete(prompt, system=SYSTEM_PROMPT)
    return {
        "question": question,
        "answer": answer,
        "sources": [{"source": c["source"], "heading": c["heading"], "url": c["url"]}
                    for c in chunks],
        "retrieval_method": method,
        "query_rewriting": use_query_rewriting,
        "reranking": use_reranking,
    }


if __name__ == "__main__":
    result = answer_question("How do I add a path parameter with type validation?")
    print("Q:", result["question"])
    print("\nA:", result["answer"])
    print("\nSources:")
    for s in result["sources"]:
        print(" -", s["source"], "|", s["heading"])
