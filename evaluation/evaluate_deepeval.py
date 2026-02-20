# evaluation/evaluate_deepeval.py
"""
DeepEval G-Eval custom metrics: Actionability and Correctness.

Compares:
  RAG    = hybrid retrieval + Qwen LLM
  NO-RAG = Qwen LLM with no document retrieval

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

from langchain_ollama import ChatOllama
from deepeval.models.base_model import DeepEvalBaseLLM
from deepeval.metrics import GEval
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
JUDGE_MODEL = "qwen2.5:1.5b-instruct"  # swap to qwen2.5:7b if scores are unstable
KB_PATH = os.path.join(_ROOT, "test_docs")
CHROMA_DIR = os.path.join(_ROOT, "chroma_db")
RESULTS_DIR = os.path.join(_HERE, "results")
RESULTS_FILE = os.path.join(RESULTS_DIR, "deepeval_results.json")


# ---------------------------------------------------------------------------
# Local Ollama judge — overrides DeepEval's default OpenAI judge
# ---------------------------------------------------------------------------
class OllamaJudge(DeepEvalBaseLLM):
    def __init__(self, model_name: str = JUDGE_MODEL):
        self.model_name = model_name
        self.model = ChatOllama(model=model_name, temperature=0)

    def load_model(self):
        return self.model

    def generate(self, prompt: str) -> str:
        return self.model.invoke(prompt).content

    async def a_generate(self, prompt: str) -> str:
        return self.generate(prompt)

    def get_model_name(self) -> str:
        return f"Ollama/{self.model_name}"


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
    print(f"Setting up local judge: {JUDGE_MODEL}")
    judge = OllamaJudge(model_name=JUDGE_MODEL)

    # G-Eval metrics (criteria come from eval_config.py)
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
            question, vectordb, bm25, use_sensors=False, use_hybrid=True
        )
        rag_time = time.time() - t0

        # NO-RAG answer
        t0 = time.time()
        norag_result = get_norag_answer(question)
        norag_time = time.time() - t0

        # Score RAG
        rag_test_case = LLMTestCase(input=question, actual_output=rag_result["answer"])
        actionability_metric.measure(rag_test_case)
        rag_action_score = actionability_metric.score

        correctness_metric.measure(rag_test_case)
        rag_correct_score = correctness_metric.score

        # Score NO-RAG
        norag_test_case = LLMTestCase(input=question, actual_output=norag_result["answer"])
        actionability_metric.measure(norag_test_case)
        norag_action_score = actionability_metric.score

        correctness_metric.measure(norag_test_case)
        norag_correct_score = correctness_metric.score

        per_question.append({
            "question": question,
            "category": tc["category"],
            "rag": {
                "answer": rag_result["answer"],
                "actionability": rag_action_score,
                "correctness": rag_correct_score,
                "latency": round(rag_time, 3),
            },
            "norag": {
                "answer": norag_result["answer"],
                "actionability": norag_action_score,
                "correctness": norag_correct_score,
                "latency": round(norag_time, 3),
            },
        })

        print(
            f"  RAG    — actionability: {rag_action_score:.2f}  correctness: {rag_correct_score:.2f}"
        )
        print(
            f"  NO-RAG — actionability: {norag_action_score:.2f}  correctness: {norag_correct_score:.2f}"
        )

    # -------------------------------------------------------------------------
    # Compute averages
    # -------------------------------------------------------------------------
    n = len(per_question)
    avg_rag_action = sum(r["rag"]["actionability"] for r in per_question) / n
    avg_norag_action = sum(r["norag"]["actionability"] for r in per_question) / n
    avg_rag_correct = sum(r["rag"]["correctness"] for r in per_question) / n
    avg_norag_correct = sum(r["norag"]["correctness"] for r in per_question) / n

    # -------------------------------------------------------------------------
    # Print summary comparison
    # -------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("DEEPEVAL RESULTS — RAG vs NO-RAG")
    print("=" * 60)
    print(f"{'METRIC':<25}  {'RAG':>7}  {'NO-RAG':>7}  {'DIFF':>7}")
    print("-" * 60)
    print(
        f"{'Actionability (avg)':<25}  {avg_rag_action:>7.4f}  "
        f"{avg_norag_action:>7.4f}  {avg_rag_action - avg_norag_action:>+7.4f}"
    )
    print(
        f"{'Correctness (avg)':<25}  {avg_rag_correct:>7.4f}  "
        f"{avg_norag_correct:>7.4f}  {avg_rag_correct - avg_norag_correct:>+7.4f}"
    )
    print("=" * 60)

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
                "actionability": avg_rag_action,
                "correctness": avg_rag_correct,
            },
            "norag": {
                "actionability": avg_norag_action,
                "correctness": avg_norag_correct,
            },
        },
        "per_question": per_question,
    }

    with open(RESULTS_FILE, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\nResults saved to: {RESULTS_FILE}")


if __name__ == "__main__":
    main()
