"""
Builds a ground-truth evaluation set: for a sample of chunks, generate
a question that chunk should be able to answer. This gives us
(question -> chunk_id) pairs to measure retrieval quality against.

Two modes:
- If an LLM API key is configured, use the LLM to write a natural,
  paraphrased question per chunk (the approach taught in the course --
  produces harder, more realistic queries because they don't just repeat
  the chunk's vocabulary).
- Otherwise (demo mode), fall back to a deterministic heuristic so the
  evaluation script still runs end-to-end with zero setup.

Run:
    python -m app.generate_ground_truth
"""

from __future__ import annotations

import json
import os
import random
import re
from pathlib import Path

from app.llm import complete, _detect_provider

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
CHUNKS_FILE = DATA_DIR / "chunks.json"
GROUND_TRUTH_FILE = DATA_DIR / "ground_truth.json"

SAMPLE_SIZE = 120  # keep the eval set small enough to run quickly / cheaply
QUESTIONS_PER_CHUNK = 2

LLM_PROMPT = """You will be given an excerpt from the FastAPI documentation.
Write {n} short, natural questions a developer might ask that this excerpt
answers. Do not copy phrases verbatim from the text -- paraphrase.
Return ONLY the questions, one per line, no numbering.

EXCERPT ({heading}):
{text}
""".strip()


def heuristic_question(chunk: dict) -> str:
    heading = re.sub(r"[`*_]", "", chunk["heading"]).strip()
    heading = heading[0].lower() + heading[1:] if heading else "this"
    templates = [
        f"How does FastAPI handle {heading}?",
        f"What do I need to know about {heading} in FastAPI?",
        f"Can you explain {heading} in FastAPI?",
    ]
    return random.choice(templates)


def build_ground_truth():
    with open(CHUNKS_FILE) as f:
        chunks = json.load(f)

    random.seed(42)
    sample = random.sample(chunks, min(SAMPLE_SIZE, len(chunks)))

    use_llm = _detect_provider() != "demo"
    print(f"Generating ground truth for {len(sample)} chunks "
          f"(mode: {'LLM' if use_llm else 'heuristic/demo'})...")

    pairs = []
    for chunk in sample:
        if use_llm:
            try:
                prompt = LLM_PROMPT.format(
                    n=QUESTIONS_PER_CHUNK, heading=chunk["heading"], text=chunk["text"][:800]
                )
                raw = complete(prompt)
                questions = [q.strip("- ").strip() for q in raw.split("\n") if q.strip()]
            except Exception as e:
                print(f"  [WARN] LLM question-gen failed for {chunk['id']}: {e}")
                questions = [heuristic_question(chunk)]
        else:
            questions = [heuristic_question(chunk) for _ in range(QUESTIONS_PER_CHUNK)]

        for q in questions:
            pairs.append({"question": q, "chunk_id": chunk["id"], "source": chunk["source"]})

    with open(GROUND_TRUTH_FILE, "w") as f:
        json.dump(pairs, f, indent=2)

    print(f"Wrote {len(pairs)} question/chunk pairs -> {GROUND_TRUTH_FILE}")


if __name__ == "__main__":
    build_ground_truth()
