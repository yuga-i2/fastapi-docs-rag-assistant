"""
LLM-output evaluation: compares two different generation approaches
(prompt variants) using an LLM-as-a-judge, so we can say which one
produces better-grounded, more relevant answers -- not just which
retrieval is best, but which *generation* setup is best.

Variant A ("concise"): terse system prompt, answer only.
Variant B ("grounded+cited"): system prompt requires explicit source
citation and an explicit "not in docs" fallback -- stricter grounding,
usually fewer hallucinations, slightly longer answers.

The judge scores each answer 1-5 on: relevance, groundedness (is it
actually supported by the retrieved context, not made up).

Run:
    python -m app.evaluate_llm
"""

from __future__ import annotations

import json
import random
import re
import time
from pathlib import Path

from app.llm import complete, _detect_provider
from app.search import get_index
from app.rag import build_context

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
GROUND_TRUTH_FILE = DATA_DIR / "ground_truth.json"
RESULTS_FILE = DATA_DIR / "llm_eval_results.json"

# Free-tier LLM APIs (e.g. Groq) enforce tight tokens-per-minute limits.
# Keep the sample small and pace requests so a full run doesn't 429.
SAMPLE_SIZE = 8
SECONDS_BETWEEN_CALLS = 3
MAX_RETRIES = 5


def complete_with_retry(prompt: str, system: str = "") -> str:
    """Wraps complete() with backoff on rate-limit errors, so a slow
    free-tier API (e.g. Groq's 8k TPM cap) doesn't crash a batch eval run."""
    delay = 2.0
    for attempt in range(MAX_RETRIES):
        try:
            return complete(prompt, system=system)
        except Exception as e:
            is_rate_limit = "429" in str(e) or "rate_limit" in str(e).lower()
            if is_rate_limit and attempt < MAX_RETRIES - 1:
                print(f"  rate limited, waiting {delay:.0f}s before retry...")
                time.sleep(delay)
                delay *= 2
            else:
                raise

VARIANTS = {
    "concise": {
        "system": "Answer the question using only the CONTEXT. Be concise.",
    },
    "grounded_cited": {
        "system": (
            "Answer the question using ONLY the CONTEXT. If the CONTEXT does "
            "not contain the answer, say 'Not covered in the docs I have.' "
            "Cite the source file for every claim you make."
        ),
    },
}

JUDGE_PROMPT = """You are grading an AI assistant's answer to a question
about FastAPI, given the documentation CONTEXT it had access to.

QUESTION: {question}

CONTEXT:
{context}

ANSWER TO GRADE:
{answer}

Score the ANSWER from 1 (bad) to 5 (excellent) on two axes:
- relevance: does it actually address the question
- groundedness: is every claim supported by the CONTEXT (no invented facts)

Respond with ONLY a JSON object like:
{{"relevance": 4, "groundedness": 5}}
""".strip()


def heuristic_judge(answer: str, context: str) -> dict:
    """Fallback scorer used in demo mode (no LLM key): a crude but
    deterministic proxy -- word overlap between answer and context as a
    stand-in for groundedness, and answer length as a proxy for effort/
    relevance. Not a substitute for a real LLM judge; documented as such."""
    ans_words = set(re.findall(r"[a-z0-9]+", answer.lower()))
    ctx_words = set(re.findall(r"[a-z0-9]+", context.lower()))
    overlap = len(ans_words & ctx_words) / max(len(ans_words), 1)
    groundedness = round(1 + overlap * 4, 1)
    relevance = round(min(5, 2 + len(answer) / 200), 1)
    return {"relevance": relevance, "groundedness": groundedness}


def judge(question: str, context: str, answer: str) -> dict:
    if _detect_provider() == "demo":
        return heuristic_judge(answer, context)
    try:
        raw = complete_with_retry(JUDGE_PROMPT.format(question=question, context=context[:3000], answer=answer))
        match = re.search(r"\{.*\}", raw, flags=re.DOTALL)
        return json.loads(match.group(0)) if match else heuristic_judge(answer, context)
    except Exception:
        return heuristic_judge(answer, context)


def run():
    with open(GROUND_TRUTH_FILE) as f:
        ground_truth = json.load(f)
    random.seed(7)
    sample = random.sample(ground_truth, min(SAMPLE_SIZE, len(ground_truth)))

    index = get_index()
    results = {name: {"relevance": [], "groundedness": []} for name in VARIANTS}

    for i, pair in enumerate(sample):
        print(f"[{i+1}/{len(sample)}] {pair['question'][:70]}")
        chunks = index.search(pair["question"], top_k=5, method="hybrid")
        context = build_context(chunks)

        for name, cfg in VARIANTS.items():
            prompt = f"QUESTION: {pair['question']}\n\nCONTEXT:\n{context}"
            answer = complete_with_retry(prompt, system=cfg["system"])
            time.sleep(SECONDS_BETWEEN_CALLS)
            scores = judge(pair["question"], context, answer)
            time.sleep(SECONDS_BETWEEN_CALLS)
            results[name]["relevance"].append(scores["relevance"])
            results[name]["groundedness"].append(scores["groundedness"])

    summary = {}
    for name, scores in results.items():
        summary[name] = {
            "avg_relevance": round(sum(scores["relevance"]) / len(scores["relevance"]), 2),
            "avg_groundedness": round(sum(scores["groundedness"]) / len(scores["groundedness"]), 2),
            "n": len(scores["relevance"]),
        }

    best = max(summary, key=lambda k: summary[k]["avg_relevance"] + summary[k]["avg_groundedness"])
    output = {"variants": summary, "best_variant": best,
              "judge_mode": "llm" if _detect_provider() != "demo" else "heuristic_fallback"}

    with open(RESULTS_FILE, "w") as f:
        json.dump(output, f, indent=2)

    for name, s in summary.items():
        print(f"{name}: relevance={s['avg_relevance']}  groundedness={s['avg_groundedness']}")
    print(f"\nBest variant: {best}")
    print(f"(judge mode: {output['judge_mode']})")
    print(f"Results written to {RESULTS_FILE}")
    return output


if __name__ == "__main__":
    run()
