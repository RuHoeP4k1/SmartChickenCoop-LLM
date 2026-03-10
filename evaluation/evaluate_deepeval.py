# evaluation/evaluate_deepeval.py
"""
DeepEval G-Eval custom metrics: Actionability and Correctness.

Compares:
  RAG    = hybrid retrieval + Qwen/smoll LLM
  NO-RAG = Qwen/smoll LLM with no document retrieval

Criteria come from eval_config.py — a teammate must fill these in before
the scores are meaningful.

NOTE: Scores will be meaningless until a teammate has filled in eval_config.py.
      Check that ACTIONABILITY_CRITERIA and CORRECTNESS_CRITERIA do not contain
      "FILL_IN" before trusting the output of this script.
"""

import os
import sys

# Disable DeepEval telemetry before any other deepeval import
os.environ["DEEPEVAL_TELEMETRY_OPT_OUT"] = "YES"

import json
import time
from datetime import datetime

# =============================================================================
# SCOPE: SENSOR-INDEPENDENT EVALUATION
# This script evaluates RAG knowledge quality only. Sensor data is explicitly
# disabled (use_sensors=False) on every answer_query() call.
# Reason: mixing live sensor readings into this evaluation would conflate two
# separate questions — "does RAG improve knowledge?" vs "does the system respond
# correctly to real coop conditions?" The second question is answered separately
# when the physical test setup is operational.
# =============================================================================

# =============================================================================
# FUTURE: STRONGER MODEL BENCHMARK
# To compare against a stronger model (e.g. Claude API, GPT-4), add a third
# condition here that calls the external API. Do NOT implement this yet —
# discuss with supervisor whether we want this comparison in the final report.
# When approved, replace this comment with the implementation.
# =============================================================================

# ---------------------------------------------------------------------------
# Path setup — makes script runnable from any directory
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _ROOT)

from dotenv import load_dotenv
load_dotenv(dotenv_path=os.path.join(_ROOT, ".env"))

from langchain_anthropic import ChatAnthropic
from deepeval.models.base_model import DeepEvalBaseLLM
from deepeval.metrics import GEval, FaithfulnessMetric, AnswerRelevancyMetric
from deepeval.test_case import LLMTestCase, LLMTestCaseParams

from rag_functions import (
    load_documents, split_documents, build_vector_store,
    build_bm25_retriever, answer_query,
)
from evaluate_rag import get_norag_answer
from evaluation_data import TEST_CASES
from eval_config import ACTIONABILITY_CRITERIA, CORRECTNESS_CRITERIA

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
JUDGE_MODEL = os.getenv("CLAUDE_JUDGE_MODEL", "claude-haiku-4-5-20251001")
KB_PATH = os.path.join(_ROOT, "test_docs")
CHROMA_DIR = os.path.join(_ROOT, "chroma_db")
RESULTS_DIR = os.path.join(_HERE, "results")
RESULTS_FILE = os.path.join(RESULTS_DIR, "deepeval_results.json")


# ---------------------------------------------------------------------------
# Claude Haiku judge — overrides DeepEval's default OpenAI judge
# Requires ANTHROPIC_API_KEY in .env
# ---------------------------------------------------------------------------
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


def _check_fill_in():
    """Warn if criteria are still placeholder text."""
    issues = []
    if "FILL_IN" in ACTIONABILITY_CRITERIA:
        issues.append("ACTIONABILITY_CRITERIA")
    if "FILL_IN" in CORRECTNESS_CRITERIA:
        issues.append("CORRECTNESS_CRITERIA")
    if issues:
        print("\n⚠️  WARNING: The following criteria still contain placeholder text:")
        for item in issues:
            print(f"   - {item} in eval_config.py")
        print("   Scores will be meaningless until a teammate fills these in.\n")


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    _check_fill_in()

    # -------------------------------------------------------------------------
    # Judge setup — local Ollama, NOT OpenAI
    # -------------------------------------------------------------------------
    print(f"Judge: {JUDGE_MODEL}")
    judge = ClaudeJudge(model_name=JUDGE_MODEL)

    # Custom G-Eval metrics (1–3 scale, domain-specific criteria)
    actionability_metric = GEval(
        name="Actionability",
        criteria=ACTIONABILITY_CRITERIA,
        evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
        model=judge,
    )
    correctness_metric = GEval(
        name="Correctness",
        criteria=CORRECTNESS_CRITERIA,
        evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
        model=judge,
    )

    # Standard DeepEval RAG metrics (0–1 scale)
    faithfulness_metric     = FaithfulnessMetric(model=judge)
    answer_relevancy_metric = AnswerRelevancyMetric(model=judge)

    # -------------------------------------------------------------------------
    # RAG pipeline setup
    # -------------------------------------------------------------------------
    print("Setting up RAG pipeline...")
    docs = load_documents(KB_PATH)
    chunks = split_documents(docs)
    vectordb = build_vector_store(chunks, persist_dir=CHROMA_DIR, folder_path=KB_PATH)
    bm25 = build_bm25_retriever(chunks)
    print("RAG pipeline ready\n")

    # -------------------------------------------------------------------------
    # Evaluate each question
    # -------------------------------------------------------------------------
    per_question = []

    for i, tc in enumerate(TEST_CASES, 1):
        question = tc["question"]
        print(f"[{i}/{len(TEST_CASES)}] {question}")

        # RAG answer
        t0 = time.time()
        rag_result = answer_query(
            question, vectordb, bm25, use_sensors=False, use_hybrid=True,
        )
        rag_time = time.time() - t0

        # NO-RAG answer
        t0 = time.time()
        norag_result = get_norag_answer(question)
        norag_time = time.time() - t0

        gt      = tc.get("ground_truth", "")
        ret_ctx = [doc.page_content for doc in rag_result["documents"]]

        # Score RAG — full context: retrieval_context enables RAG-specific metrics
        rag_test_case = LLMTestCase(
            input=question,
            actual_output=rag_result["answer"],
            expected_output=gt,
            retrieval_context=ret_ctx,
        )
        actionability_metric.measure(rag_test_case)
        rag_action_score = actionability_metric.score
        correctness_metric.measure(rag_test_case)
        rag_correct_score = correctness_metric.score
        faithfulness_metric.measure(rag_test_case)
        rag_faith_score = faithfulness_metric.score
        answer_relevancy_metric.measure(rag_test_case)
        rag_rel_score = answer_relevancy_metric.score

        # Score NO-RAG — no retrieval_context (no documents were retrieved)
        norag_test_case = LLMTestCase(
            input=question,
            actual_output=norag_result["answer"],
            expected_output=gt,
            retrieval_context=[],
        )
        actionability_metric.measure(norag_test_case)
        norag_action_score = actionability_metric.score
        correctness_metric.measure(norag_test_case)
        norag_correct_score = correctness_metric.score
        faithfulness_metric.measure(norag_test_case)
        norag_faith_score = faithfulness_metric.score
        answer_relevancy_metric.measure(norag_test_case)
        norag_rel_score = answer_relevancy_metric.score

        per_question.append({
            "question": question,
            "category": tc["category"],
            "rag": {
                "answer": rag_result["answer"],
                "actionability":    rag_action_score,
                "correctness":      rag_correct_score,
                "faithfulness":     rag_faith_score,
                "answer_relevancy": rag_rel_score,
                "latency": round(rag_time, 3),
            },
            "norag": {
                "answer": norag_result["answer"],
                "actionability":    norag_action_score,
                "correctness":      norag_correct_score,
                "faithfulness":     norag_faith_score,
                "answer_relevancy": norag_rel_score,
                "latency": round(norag_time, 3),
            },
        })

        print(
            f"  RAG    — action: {rag_action_score:.2f}  correct: {rag_correct_score:.2f}"
            f"  faithful: {rag_faith_score:.2f}  relevancy: {rag_rel_score:.2f}"
        )
        print(
            f"  NO-RAG — action: {norag_action_score:.2f}  correct: {norag_correct_score:.2f}"
            f"  faithful: {norag_faith_score:.2f}  relevancy: {norag_rel_score:.2f}"
        )

    # -------------------------------------------------------------------------
    # Compute averages
    # -------------------------------------------------------------------------
    n = len(per_question)

    def _avg(cond, key): return sum(r[cond][key] for r in per_question) / n

    avg_rag_action    = _avg("rag",   "actionability")
    avg_norag_action  = _avg("norag", "actionability")
    avg_rag_correct   = _avg("rag",   "correctness")
    avg_norag_correct = _avg("norag", "correctness")
    avg_rag_faith     = _avg("rag",   "faithfulness")
    avg_norag_faith   = _avg("norag", "faithfulness")
    avg_rag_rel       = _avg("rag",   "answer_relevancy")
    avg_norag_rel     = _avg("norag", "answer_relevancy")

    # -------------------------------------------------------------------------
    # Print summary comparison
    # -------------------------------------------------------------------------
    print("\n" + "=" * 65)
    print("DEEPEVAL RESULTS — RAG vs NO-RAG")
    print("=" * 65)
    print(f"{'METRIC':<28}  {'RAG':>7}  {'NO-RAG':>7}  {'DIFF':>7}")
    print("-" * 65)
    rows = [
        ("Actionability (1–3)",  avg_rag_action,   avg_norag_action),
        ("Correctness (1–3)",    avg_rag_correct,  avg_norag_correct),
        ("Faithfulness (0–1)",   avg_rag_faith,    avg_norag_faith),
        ("Answer Relevancy (0–1)", avg_rag_rel,    avg_norag_rel),
    ]
    for label, rag_val, norag_val in rows:
        print(f"{label:<28}  {rag_val:>7.4f}  {norag_val:>7.4f}  {rag_val - norag_val:>+7.4f}")
    print("=" * 65)

    # -------------------------------------------------------------------------
    # Save results
    # -------------------------------------------------------------------------
    output = {
        "timestamp": datetime.now().isoformat(),
        "judge_model": JUDGE_MODEL,
        "criteria_filled": "FILL_IN" not in ACTIONABILITY_CRITERIA
        and "FILL_IN" not in CORRECTNESS_CRITERIA,
        "n_questions": n,
        "averages": {
            "rag": {
                "actionability":    avg_rag_action,
                "correctness":      avg_rag_correct,
                "faithfulness":     avg_rag_faith,
                "answer_relevancy": avg_rag_rel,
            },
            "norag": {
                "actionability":    avg_norag_action,
                "correctness":      avg_norag_correct,
                "faithfulness":     avg_norag_faith,
                "answer_relevancy": avg_norag_rel,
            },
        },
        "per_question": per_question,
    }

    with open(RESULTS_FILE, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\nResults saved to: {RESULTS_FILE}")


if __name__ == "__main__":
    main()
