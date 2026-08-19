"""
Retrieval evaluation: compares keyword-only, vector-only, and hybrid
search using Hit Rate@k and Mean Reciprocal Rank (MRR), against the
ground-truth set produced by generate_ground_truth.py.

This is what lets us say "we evaluated multiple retrieval approaches and
picked the best one" with actual numbers, not a guess.

Run:
    python -m app.evaluate_retrieval
"""

from __future__ import annotations

import json
from pathlib import Path

from app.search import get_index

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
GROUND_TRUTH_FILE = DATA_DIR / "ground_truth.json"
RESULTS_FILE = DATA_DIR / "retrieval_eval_results.json"

TOP_K = 5


def hit_rate(relevance: list[list[bool]]) -> float:
    return sum(any(row) for row in relevance) / len(relevance)


def mrr(relevance: list[list[bool]]) -> float:
    total = 0.0
    for row in relevance:
        for rank, is_relevant in enumerate(row, start=1):
            if is_relevant:
                total += 1.0 / rank
                break
    return total / len(relevance)


def evaluate_method(index, ground_truth: list[dict], method: str, top_k: int = TOP_K) -> dict:
    relevance = []
    for pair in ground_truth:
        results = index.search(pair["question"], top_k=top_k, method=method)
        result_ids = [r["id"] for r in results]
        relevance.append([rid == pair["chunk_id"] for rid in result_ids])
    return {
        "method": method,
        "hit_rate": round(hit_rate(relevance), 4),
        "mrr": round(mrr(relevance), 4),
        "n_questions": len(ground_truth),
        "top_k": top_k,
    }


def run():
    with open(GROUND_TRUTH_FILE) as f:
        ground_truth = json.load(f)

    index = get_index()

    results = []
    for method in ["keyword", "vector", "hybrid"]:
        print(f"Evaluating method: {method} ...")
        res = evaluate_method(index, ground_truth, method)
        results.append(res)
        print(f"  hit_rate@{TOP_K}={res['hit_rate']}   mrr={res['mrr']}")

    best = max(results, key=lambda r: r["hit_rate"])
    summary = {"results": results, "best_method": best["method"]}

    with open(RESULTS_FILE, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nBest method by hit_rate: {best['method']}")
    print(f"Results written to {RESULTS_FILE}")
    return summary


if __name__ == "__main__":
    run()
