"""
evaluate_prompt_variants.py — Prompt design evaluation

Tests different SIMPLE_PROMPT variants against the same fixed RAG pipeline.
Run AFTER sweep.py has determined the best model / chunk_size / k / weights.

The goal: find which prompt template produces the best answers for general
(no-sensor) questions, measured both by automated heuristics and by
human pairwise preference.

Two output modes:
  1. Automated heuristic scoring (topic coverage, length, actionability)
  2. --export-pairs  → writes results/prompt_pairs.json for human_ranking.py

Usage:
    python evaluation/evaluate_prompt_variants.py
    python evaluation/evaluate_prompt_variants.py --export-pairs
    python evaluation/evaluate_prompt_variants.py --n-questions 5   # smoke test

Design axes tested across the four variants:
  baseline  — current production SIMPLE_PROMPT
  structured — forces numbered output (What to do / Call a vet if)
  concise    — minimal instructions, relies on the model's own judgment
  expert     — positions the assistant as a poultry scientist
"""

import os
import sys
import json
import time
import argparse
import random
from typing import Dict, List

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _ROOT)
sys.path.insert(0, _HERE)

from dotenv import load_dotenv
load_dotenv(dotenv_path=os.path.join(_ROOT, ".env"))

from backend.rag_functions import (
    load_documents, split_documents, build_vector_store,
    build_bm25_retriever, hybrid_search, semantic_search,
    format_context, generate_response,
)
from evaluate_rag import evaluate_answer_quality
from evaluation_data import TEST_CASES


# =============================================================================
# PROMPT VARIANTS
# Each variant is a format string with {context} and {query} placeholders.
# All use the same RAG-retrieved context; only the prompt framing changes.
# =============================================================================

PROMPT_VARIANTS: Dict[str, str] = {

    # -------------------------------------------------------------------------
    # BASELINE — current production prompt (from backend/prompts.py SIMPLE_PROMPT)
    # -------------------------------------------------------------------------
    "baseline": """You are a friendly, knowledgeable assistant for hobby chicken keepers.
Keep answers practical and clear. Use plain language, no jargon.
Base your answer on the provided knowledge. If it doesn't fully cover the question, share the most useful practical advice you can — only admit uncertainty if you have no relevant knowledge at all.
Never suggest medications, dosages, or chemical treatments.
Always use metric units (°C, kg, cm, litres) — convert any imperial values from the knowledge base before answering.
Answer directly. Do not reproduce source labels, XML tags, or knowledge base structure in your answer.

---
{context}
---

Question: {query}

Answer helpfully and concisely based on the knowledge above.
If this involves a sick or injured chicken, mention when a vet or experienced keeper should be contacted.
Keep it short — a few sentences or a brief list is usually enough.""",

    # -------------------------------------------------------------------------
    # STRUCTURED — forces explicit output sections
    # Similar to the no-RAG baseline in evaluate_rag.py; tests if structure helps.
    # -------------------------------------------------------------------------
    "structured": """You are ChickenCare AI — a practical assistant for hobby chicken keepers.

Hard rules:
- NEVER suggest medications, dosages, or chemical treatments.
- Use plain, beginner-friendly language.
- Use metric units (°C, kg, cm, litres).

---
{context}
---

Question: {query}

Answer in this format:

**Short answer:** (1–2 sentences — direct and specific)

**What to do:**
1. [Specific step]
2. [Second step]
3. [What to monitor / for how long]
(Write "No action needed" if the question is purely factual.)

**Call a vet if:** [1–2 specific red flags. Write "Not applicable" if unrelated to health.]""",

    # -------------------------------------------------------------------------
    # CONCISE — minimal framing, relies on the model's instruction-following
    # Tests whether shorter prompts produce better or worse output.
    # -------------------------------------------------------------------------
    "concise": """You are a helpful assistant for backyard chicken keepers. Answer based only on the knowledge below. Be concise and practical. Use metric units.

---
{context}
---

Question: {query}

Answer:""",

    # -------------------------------------------------------------------------
    # EXPERT — positions assistant as a poultry scientist
    # Tests whether authoritative framing improves factual accuracy/completeness.
    # -------------------------------------------------------------------------
    "expert": """You are a poultry scientist advising hobby chicken keepers. You give accurate, evidence-based advice grounded in the knowledge below. You never suggest medications or chemical treatments. You use metric units. You are direct and practical — not overly cautious.

---
{context}
---

Question: {query}

Provide a clear, practical answer based on the knowledge above. If this involves a sick or injured bird, note when professional advice is warranted. Avoid unnecessary hedging.""",
}


# =============================================================================
# RUNNER
# =============================================================================

def run_variant(
    variant_name: str,
    prompt_template: str,
    questions: List[Dict],
    vectordb,
    bm25_retriever,
    k: int = 4,
    use_hybrid: bool = True,
    weights: list = None,
    llm_model: str = None,
) -> List[Dict]:
    """Run one prompt variant over all questions and return scored results."""
    results = []

    for i, test in enumerate(questions, 1):
        question = test["question"]
        category = test["category"]
        expected_topics = test["expected_topics"]

        # Retrieve context (same pipeline as production)
        if use_hybrid and bm25_retriever is not None:
            docs = hybrid_search(vectordb, bm25_retriever, question, k=k, weights=weights)
        else:
            docs = semantic_search(vectordb, question, k=k)
        context = format_context(docs)

        # Build prompt from this variant's template
        prompt = prompt_template.format(context=context, query=question)

        # Generate answer
        t0 = time.time()
        answer = generate_response(prompt, model=llm_model)
        elapsed = time.time() - t0

        quality = evaluate_answer_quality(answer, expected_topics, category)

        results.append({
            "variant": variant_name,
            "question": question,
            "category": category,
            "answer": answer,
            "time": round(elapsed, 2),
            "quality": quality,
        })

        print(f"  [{i}/{len(questions)}] {category}: overall={quality['overall']:.1f}  "
              f"topics={quality['topics_found']}/{len(expected_topics)}  {elapsed:.1f}s")

    return results


def run_evaluation(
    n_questions: int = None,
    export_pairs: bool = False,
    k: int = 4,
    use_hybrid: bool = True,
    weights: list = None,
    llm_model: str = None,
    chunk_size: int = 600,
):
    KB_PATH    = os.path.join(_ROOT, "test_docs")
    CHROMA_DIR = os.path.join(_ROOT, "chroma_db")
    RESULTS_DIR = os.path.join(_HERE, "results")
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("=" * 80)
    print("PROMPT VARIANT EVALUATION")
    print(f"  Model: {llm_model or os.getenv('OLLAMA_MODEL', 'smollm2:1.7b')}")
    print(f"  k={k}  hybrid={use_hybrid}  chunk_size={chunk_size}")
    print("=" * 80)

    print("\nBuilding RAG pipeline...")
    docs    = load_documents(KB_PATH)
    chunks  = split_documents(docs, chunk_size=chunk_size)
    vectordb = build_vector_store(chunks, persist_dir=CHROMA_DIR, folder_path=KB_PATH)
    bm25    = build_bm25_retriever(chunks) if use_hybrid else None
    print("RAG pipeline ready.\n")

    questions = TEST_CASES[:n_questions] if n_questions else TEST_CASES

    all_results: Dict[str, List[Dict]] = {}

    for variant_name, prompt_template in PROMPT_VARIANTS.items():
        print(f"\n{'─' * 60}")
        print(f"Variant: {variant_name.upper()}")
        print(f"{'─' * 60}")
        all_results[variant_name] = run_variant(
            variant_name, prompt_template, questions,
            vectordb, bm25, k=k, use_hybrid=use_hybrid,
            weights=weights, llm_model=llm_model,
        )

    # ── Summary table ────────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"{'Variant':<14} {'Avg overall':>11} {'Avg topics%':>11} {'Avg length':>10} {'Avg action':>10}")
    print("─" * 60)
    for variant_name, results in all_results.items():
        n = len(results)
        avg_overall  = sum(r["quality"]["overall"]          for r in results) / n
        avg_topics   = sum(r["quality"]["topic_coverage"]   for r in results) / n
        avg_length   = sum(r["quality"]["length_appropriate"] for r in results) / n
        avg_action   = sum(r["quality"]["actionable"]       for r in results) / n
        print(f"{variant_name:<14} {avg_overall:>11.1f} {avg_topics:>11.1f} {avg_length:>10.1f} {avg_action:>10.1f}")

    # ── Save raw results ──────────────────────────────────────────────────────
    out_path = os.path.join(RESULTS_DIR, "prompt_variant_results.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to: {out_path}")

    # ── Export pairwise comparison data for human_ranking.py ─────────────────
    if export_pairs:
        pairs = _build_pairwise_data(all_results, questions)
        pairs_path = os.path.join(RESULTS_DIR, "prompt_pairs.json")
        with open(pairs_path, "w") as f:
            json.dump(pairs, f, indent=2)
        print(f"Pairwise data saved to: {pairs_path}")
        print(f"  {len(pairs)} comparisons ready for human_ranking.py")


def _build_pairwise_data(all_results: Dict, questions: List[Dict]) -> List[Dict]:
    """
    Build all (question, variant_A_answer, variant_B_answer) pairs for
    human pairwise preference evaluation.

    Each pair is stored with variant names shuffled so raters can't detect
    which variant is A or B (blind comparison).
    """
    variant_names = list(all_results.keys())
    pairs = []

    for qi, test in enumerate(questions):
        question = test["question"]
        category = test["category"]

        # All unordered pairs of variants
        for i in range(len(variant_names)):
            for j in range(i + 1, len(variant_names)):
                v_a = variant_names[i]
                v_b = variant_names[j]

                answer_a = all_results[v_a][qi]["answer"]
                answer_b = all_results[v_b][qi]["answer"]

                # Randomly swap A/B so raters are blind to which variant they're seeing
                if random.random() < 0.5:
                    v_a, v_b = v_b, v_a
                    answer_a, answer_b = answer_b, answer_a

                pairs.append({
                    "pair_id": f"q{qi:03d}_{v_a}_vs_{v_b}",
                    "question": question,
                    "category": category,
                    "answer_a": {"variant": v_a, "text": answer_a},
                    "answer_b": {"variant": v_b, "text": answer_b},
                    "winner": None,  # filled in by human_ranking.py
                })

    random.shuffle(pairs)
    return pairs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prompt variant evaluation")
    parser.add_argument("--n-questions", type=int, default=None,
                        help="Limit to first N questions (smoke test)")
    parser.add_argument("--export-pairs", action="store_true",
                        help="Export pairwise JSON for human ranking")
    parser.add_argument("--k", type=int, default=4)
    parser.add_argument("--chunk-size", type=int, default=600)
    parser.add_argument("--no-hybrid", action="store_true",
                        help="Use pure semantic retrieval (skip BM25)")
    parser.add_argument("--model", type=str, default=None,
                        help="LLM model override (e.g. openrouter/qwen/qwen3-8b)")
    args = parser.parse_args()

    run_evaluation(
        n_questions=args.n_questions,
        export_pairs=args.export_pairs,
        k=args.k,
        use_hybrid=not args.no_hybrid,
        llm_model=args.model,
        chunk_size=args.chunk_size,
    )
