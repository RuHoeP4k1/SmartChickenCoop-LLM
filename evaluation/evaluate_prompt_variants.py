"""
evaluate_prompt_variants.py — Prompt design evaluation

Tests different SIMPLE_PROMPT variants against the same fixed RAG pipeline.
Run AFTER sweep.py has determined the best model / chunk_size / k / weights.

The goal: find which prompt template produces the best answers for general
(no-sensor) questions, measured by DeepEval G-Eval (Actionability + Correctness)
and optionally by human pairwise preference.

Two output modes:
  1. DeepEval G-Eval scoring (Actionability, Correctness — AI judge via Claude Haiku)
  2. --export-pairs  → writes results/prompt_pairs.json for human_ranking.py

Usage:
    python evaluation/evaluate_prompt_variants.py
    python evaluation/evaluate_prompt_variants.py --export-pairs
    python evaluation/evaluate_prompt_variants.py --n-questions 5   # smoke test

Design axes tested across the four variants:
  baseline  — current production SIMPLE_PROMPT
  structured — explicit output structure (short answer / steps / conditional vet)
  concise    — minimal instructions, relies on the model's own judgment
  expert     — positions the assistant as an experienced poultry specialist (researcher + hands-on keeper)
"""

import os
import sys
import json
import time
import argparse
import random
from typing import Dict, List

# Disable DeepEval telemetry before any other deepeval import
os.environ["DEEPEVAL_TELEMETRY_OPT_OUT"] = "YES"

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _ROOT)
sys.path.insert(0, _HERE)

from dotenv import load_dotenv
load_dotenv(dotenv_path=os.path.join(_ROOT, ".env"))

from langchain_anthropic import ChatAnthropic
from deepeval.models.base_model import DeepEvalBaseLLM
from deepeval.metrics.g_eval.utils import Rubric
from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCase, LLMTestCaseParams

from backend.rag_functions import (
    load_documents, split_documents, build_vector_store,
    build_bm25_retriever, hybrid_search, semantic_search,
    format_context, generate_response,
)
from evaluation_data import TEST_CASES
from eval_config import ACTIONABILITY_CRITERIA, CORRECTNESS_CRITERIA


# =============================================================================
# AI JUDGE — Claude Haiku (same setup as evaluate_deepeval.py)
# =============================================================================

JUDGE_MODEL = os.getenv("CLAUDE_JUDGE_MODEL", "claude-haiku-4-5-20251001")


class ClaudeJudge(DeepEvalBaseLLM):
    def __init__(self, model_name: str = JUDGE_MODEL):
        self.model_name = model_name
        self.model = ChatAnthropic(model=model_name, temperature=0)

    def load_model(self):
        return self.model

    def generate(self, prompt: str) -> str:
        return self.model.invoke(prompt).content

    async def a_generate(self, prompt: str) -> str:
        return self.generate(prompt)

    def get_model_name(self) -> str:
        return f"Claude/{self.model_name}"


# =============================================================================
# G-EVAL METRICS — Actionability + Correctness (LLM answer quality only)
# =============================================================================

_ACTION_RUBRIC = [
    Rubric(score_range=(0, 3), expected_outcome="Not actionable — vague generalities, nothing the user can act on, repeats the question."),
    Rubric(score_range=(4, 6), expected_outcome="Somewhat actionable — at least one useful piece of info or step, but leaves important gaps the user must fill."),
    Rubric(score_range=(7, 10), expected_outcome="Genuinely helpful — directly answers the question, user knows what to do next without searching further."),
]

_CORRECT_RUBRIC = [
    Rubric(score_range=(0, 3), expected_outcome="Incorrect or harmful — wrong facts, dangerous advice, contradicts poultry welfare practice."),
    Rubric(score_range=(4, 6), expected_outcome="Mostly correct — core info is right but has a meaningful inaccuracy, omission, or is overly hedged."),
    Rubric(score_range=(7, 10), expected_outcome="Correct and appropriate — accurate, safe, calibrated to question scope."),
]


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
Only describe what is explicitly shown in the current sensor readings — never present reference knowledge as something you are currently observing.
Answer directly. Do not reproduce source labels, XML tags, or knowledge base structure in your answer.

--- Reference knowledge (background information only) ---
{context}
--- End of reference knowledge ---

Question: {query}

Answer helpfully and concisely based on the knowledge above.
If this involves a sick or injured chicken, mention when a vet or experienced keeper should be contacted.
Keep it short — a few sentences or a brief list is usually enough.""",

    # -------------------------------------------------------------------------
    # STRUCTURED — tests whether explicit output structure improves actionability.
    # Same safety rules and persona undertone as baseline, but with format template.
    # -------------------------------------------------------------------------
    "structured": """You are a practical chicken-keeping advisor — the kind of knowledgeable friend every flock owner wishes they had. You cut through the noise and give keepers exactly what they need to act on, organized so nothing gets missed.

Your rules (non-negotiable):
- Never suggest medications, dosages, or chemical treatments.
- Metric units only: °C, kg, cm, litres. Convert any imperial values from the source material before answering.
- Use plain language — your reader is a hobby keeper, not a vet student.

--- Reference knowledge (background information only) ---
{context}
--- End of reference knowledge ---

Question: {query}

Give your answer in this structure — it makes the advice easier to act on:

**The short answer:** One or two sentences. Direct. No hedging.

**Steps to take:**
1. First action (specific — what exactly, not just "check the coop")
2. Second action
3. What to watch for and over what timeframe
(If the question is purely factual with nothing to act on, write "No action needed — this is background knowledge.")

**When to call a vet or experienced keeper:** Name the specific warning signs. Only include this section if the question touches health or injury — do not add it as a default safety disclaimer.""",

    # -------------------------------------------------------------------------
    # CONCISE — minimal framing, tests whether baseline's explicit guidance is
    # necessary or whether the model performs equally well with less instruction.
    # -------------------------------------------------------------------------
    "concise": """You are a knowledgeable, practical assistant for hobby chicken keepers. Give direct, plain-language advice. Never suggest medications or chemical treatments. Use metric units (°C, kg, cm, litres).

--- Reference knowledge (background information only) ---
{context}
--- End of reference knowledge ---

Question: {query}

Answer concisely based on the knowledge above. Mention a vet only if the question involves illness or injury.""",

    # -------------------------------------------------------------------------
    # EXPERT — positions assistant as experienced poultry specialist
    # Tests whether domain-expert framing (research + hands-on) improves
    # accuracy and confidence compared to general-assistant baseline.
    # -------------------------------------------------------------------------
    "expert": """You are an experienced poultry specialist — part researcher, part seasoned flock keeper. You have spent years studying poultry health and husbandry, and just as many years applying it with real backyard flocks. You know the difference between textbook advice and what actually works at 6 a.m. on a cold morning.

You give advice the way an expert gives advice to a friend: direct, accurate, grounded in evidence, and calibrated to what a hobby keeper can realistically do. You do not inflate risks, add unnecessary caveats, or recommend vets for problems an experienced keeper handles routinely.

Constraints:
- Never suggest medications, dosages, or chemical treatments.
- Use metric units (°C, kg, cm, litres). Convert any imperial figures from source material before answering.

--- Reference knowledge (background information only) ---
{context}
--- End of reference knowledge ---

Question: {query}

Answer based on the knowledge above. Be direct and complete — give the keeper enough to act confidently. Where the knowledge base leaves a gap, draw on sound general principles rather than refusing to answer. If the question involves genuine illness or injury, specify which signs would warrant professional advice.""",
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
    actionability_metric: GEval,
    correctness_metric: GEval,
    k: int = 4,
    use_hybrid: bool = False,
    weights: list = None,
    llm_model: str = None,
) -> List[Dict]:
    """Run one prompt variant over all questions and return scored results."""
    results = []

    for i, test in enumerate(questions, 1):
        question = test["question"]
        category = test["category"]

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

        # Score with DeepEval G-Eval (AI judge)
        test_case = LLMTestCase(input=question, actual_output=answer)

        actionability_metric.measure(test_case)
        action_score = actionability_metric.score

        correctness_metric.measure(test_case)
        correct_score = correctness_metric.score

        results.append({
            "variant": variant_name,
            "question": question,
            "category": category,
            "answer": answer,
            "time": round(elapsed, 2),
            "actionability": action_score,
            "correctness": correct_score,
        })

        print(f"  [{i}/{len(questions)}] {category}: action={action_score:.2f}  "
              f"correct={correct_score:.2f}  {elapsed:.1f}s")

    return results


def run_evaluation(
    n_questions: int = None,
    export_pairs: bool = False,
    k: int = 4,
    use_hybrid: bool = False,
    weights: list = None,
    llm_model: str = None,
    chunk_size: int = 1000,
):
    KB_PATH    = os.path.join(_ROOT, "test_docs")
    CHROMA_DIR = os.path.join(_ROOT, "chroma_db")
    RESULTS_DIR = os.path.join(_HERE, "results")
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("=" * 80)
    print("PROMPT VARIANT EVALUATION")
    print(f"  LLM:   {llm_model or os.getenv('OLLAMA_MODEL', 'smollm2:1.7b')}")
    print(f"  Judge: {JUDGE_MODEL}")
    print(f"  k={k}  hybrid={use_hybrid}  chunk_size={chunk_size}")
    print("=" * 80)

    # ── DeepEval judge setup ─────────────────────────────────────────────────
    print("\nSetting up AI judge...")
    judge = ClaudeJudge(model_name=JUDGE_MODEL)

    actionability_metric = GEval(
        name="Actionability",
        criteria=ACTIONABILITY_CRITERIA,
        evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
        model=judge,
        rubric=_ACTION_RUBRIC,
    )
    correctness_metric = GEval(
        name="Correctness",
        criteria=CORRECTNESS_CRITERIA,
        evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
        model=judge,
        rubric=_CORRECT_RUBRIC,
    )

    # ── RAG pipeline setup ───────────────────────────────────────────────────
    print("Building RAG pipeline...")
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
            vectordb, bm25,
            actionability_metric=actionability_metric,
            correctness_metric=correctness_metric,
            k=k, use_hybrid=use_hybrid,
            weights=weights, llm_model=llm_model,
        )

    # ── Summary table ────────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("SUMMARY  (DeepEval G-Eval, 0–1 scale)")
    print("=" * 80)
    print(f"{'Variant':<14} {'Actionability':>14} {'Correctness':>12}")
    print("─" * 44)
    for variant_name, results in all_results.items():
        n = len(results)
        avg_action  = sum(r["actionability"] for r in results) / n
        avg_correct = sum(r["correctness"]   for r in results) / n
        print(f"{variant_name:<14} {avg_action:>14.4f} {avg_correct:>12.4f}")

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
    parser.add_argument("--chunk-size", type=int, default=1000)
    parser.add_argument("--hybrid", action="store_true",
                        help="Use hybrid retrieval (BM25 + semantic) instead of pure semantic")
    parser.add_argument("--model", type=str, default=None,
                        help="LLM model override (e.g. openrouter/qwen/qwen3-8b)")
    args = parser.parse_args()

    run_evaluation(
        n_questions=args.n_questions,
        export_pairs=args.export_pairs,
        k=args.k,
        use_hybrid=args.hybrid,
        llm_model=args.model,
        chunk_size=args.chunk_size,
    )
