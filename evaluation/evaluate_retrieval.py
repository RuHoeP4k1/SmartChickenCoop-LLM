# evaluation/evaluate_retrieval.py
"""
Hybrid retrieval vs Semantic-only retrieval comparison.

Same LLM. Same prompt. Only the retrieval method changes:
  Hybrid   = 60% ChromaDB MMR + 40% BM25
  Semantic = ChromaDB MMR only

Metrics:
  1. Context Precision (RAGAS) — which method retrieves less noise?
  2. Heuristic quality score   — reuses evaluate_answer_quality from evaluate_rag.py
  3. Latency (seconds)         — is the quality tradeoff worth the speed cost?

NOTE: Context Precision requires ground_truth to be filled in evaluation_data.py.
      Until then, RAGAS scores will be meaningless (judging against "FILL_IN").

NOTE: If RAGAS scores are NaN/unstable, switch JUDGE_MODEL to qwen2.5:7b.

TARGET: ragas >= 0.2.0 (uses SingleTurnSample / EvaluationDataset API)
"""

import os
import sys
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

from langchain_ollama import ChatOllama, OllamaEmbeddings
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas import evaluate, EvaluationDataset, SingleTurnSample
from ragas.metrics import ContextPrecision
from ragas.run_config import RunConfig

from rag_functions import (
    load_documents, split_documents, build_vector_store,
    build_bm25_retriever, answer_query,
)
from evaluate_rag import evaluate_answer_quality
from evaluation_data import TEST_CASES

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
# RAGAS_JUDGE_MODEL is a separate model from the generator (OLLAMA_MODEL).
# Small models (< 3b) cannot reliably produce the structured JSON verdicts RAGAS
# requires, so scores come out NaN.  Use at least qwen2.5:7b or llama3.2:3b here.
# Pull with: ollama pull qwen2.5:7b
JUDGE_MODEL = os.getenv("RAGAS_JUDGE_MODEL", os.getenv("OLLAMA_MODEL", "qwen2.5:7b"))
KB_PATH = os.path.join(_ROOT, "test_docs")
CHROMA_DIR = os.path.join(_ROOT, "chroma_db")
RESULTS_DIR = os.path.join(_HERE, "results")
RESULTS_FILE = os.path.join(RESULTS_DIR, "retrieval_results.json")


def _avg_from_eval(eval_result, metric_name: str) -> float:
    """Extract average score for one metric from a ragas 0.4.x Results object.

    In ragas 0.4.x Results is a Pydantic model — dict() gives model fields,
    NOT metric averages. eval_result["metric_name"] returns list[float].
    """
    try:
        vals = eval_result[metric_name]
        valid = [v for v in vals if v is not None and v == v]
        return sum(valid) / len(valid) if valid else float("nan")
    except Exception:
        return float("nan")


def _fmt_score(val) -> str:
    if val != val:
        return "  N/A "
    return f"{val:.4f}"


def _fmt_time(val: float) -> str:
    return f"{val:.2f}s"


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # -------------------------------------------------------------------------
    # Judge setup — Claude Haiku (requires ANTHROPIC_API_KEY in .env)
    # -------------------------------------------------------------------------
    from langchain_anthropic import ChatAnthropic
    _claude_model = os.getenv("CLAUDE_JUDGE_MODEL", "claude-haiku-4-5-20251001")
    print(f"Judge: {_claude_model}")
    judge_llm = LangchainLLMWrapper(
        ChatAnthropic(model=_claude_model, temperature=0)
    )
    # Embeddings always come from local Ollama (ContextPrecision doesn't need embeddings,
    # but passing judge_embeddings to evaluate() is required by the API)
    judge_embeddings = LangchainEmbeddingsWrapper(
        OllamaEmbeddings(model=os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text"))
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
    # Run both conditions per question
    # -------------------------------------------------------------------------
    hybrid_samples = []
    semantic_samples = []
    per_question = []

    for i, tc in enumerate(TEST_CASES, 1):
        question = tc["question"]
        ground_truth = tc["ground_truth"]
        expected_topics = tc["expected_topics"]

        print(f"[{i}/{len(TEST_CASES)}] {question}")

        # --- Hybrid ---
        t0 = time.time()
        hybrid_result = answer_query(
            question, vectordb, bm25, use_sensors=False, use_hybrid=True,
        )
        hybrid_time = time.time() - t0
        hybrid_contexts = [doc.page_content for doc in hybrid_result["documents"]]

        # --- Semantic only ---
        t0 = time.time()
        semantic_result = answer_query(
            question, vectordb, bm25, use_sensors=False, use_hybrid=False,
        )
        semantic_time = time.time() - t0
        semantic_contexts = [doc.page_content for doc in semantic_result["documents"]]

        # --- Heuristic scores ---
        hybrid_heuristic = evaluate_answer_quality(hybrid_result["answer"], expected_topics)
        semantic_heuristic = evaluate_answer_quality(semantic_result["answer"], expected_topics)

        hybrid_samples.append(
            SingleTurnSample(
                user_input=question,
                response=hybrid_result["answer"],
                retrieved_contexts=hybrid_contexts,
                reference=ground_truth,
            )
        )
        semantic_samples.append(
            SingleTurnSample(
                user_input=question,
                response=semantic_result["answer"],
                retrieved_contexts=semantic_contexts,
                reference=ground_truth,
            )
        )

        per_question.append({
            "question": question,
            "category": tc["category"],
            "hybrid": {
                "answer": hybrid_result["answer"],
                "contexts": hybrid_contexts,
                "heuristic": hybrid_heuristic,
                "latency": round(hybrid_time, 3),
            },
            "semantic": {
                "answer": semantic_result["answer"],
                "contexts": semantic_contexts,
                "heuristic": semantic_heuristic,
                "latency": round(semantic_time, 3),
            },
        })

    # -------------------------------------------------------------------------
    # Run RAGAS ContextPrecision for both conditions
    # -------------------------------------------------------------------------
    # run_config: max_workers=1 serialises calls so local Ollama isn't overwhelmed;
    # timeout=600 gives each LLM call up to 10 minutes (qwen3:4b is slower than 1.5b).
    _run_cfg = RunConfig(timeout=600, max_workers=1)

    print("\nRunning RAGAS ContextPrecision on Hybrid condition...")
    hybrid_dataset = EvaluationDataset(samples=hybrid_samples)
    hybrid_ragas = evaluate(
        hybrid_dataset,
        metrics=[ContextPrecision(llm=judge_llm)],
        llm=judge_llm,
        embeddings=judge_embeddings,
        run_config=_run_cfg,
        raise_exceptions=False,
    )

    print("Running RAGAS ContextPrecision on Semantic condition...")
    semantic_dataset = EvaluationDataset(samples=semantic_samples)
    semantic_ragas = evaluate(
        semantic_dataset,
        metrics=[ContextPrecision(llm=judge_llm)],
        llm=judge_llm,
        embeddings=judge_embeddings,
        run_config=_run_cfg,
        raise_exceptions=False,
    )

    hybrid_cp = _avg_from_eval(hybrid_ragas, "context_precision")
    semantic_cp = _avg_from_eval(semantic_ragas, "context_precision")

    # -------------------------------------------------------------------------
    # Compute averages
    # -------------------------------------------------------------------------
    n = len(per_question)
    hybrid_avg_heuristic = sum(r["hybrid"]["heuristic"]["overall"] for r in per_question) / n
    semantic_avg_heuristic = sum(r["semantic"]["heuristic"]["overall"] for r in per_question) / n
    hybrid_avg_latency = sum(r["hybrid"]["latency"] for r in per_question) / n
    semantic_avg_latency = sum(r["semantic"]["latency"] for r in per_question) / n

    # -------------------------------------------------------------------------
    # Print per-question comparison table
    # -------------------------------------------------------------------------
    print("\n" + "=" * 78)
    print("RETRIEVAL COMPARISON — HYBRID vs SEMANTIC-ONLY")
    print("=" * 78)
    header = f"{'#':<3}  {'QUESTION':<30}  {'HYB_H':>6}  {'SEM_H':>6}  {'HYB_LAT':>8}  {'SEM_LAT':>8}"
    print(header)
    print("-" * 78)
    for i, r in enumerate(per_question, 1):
        q_short = r["question"][:30]
        hh = r["hybrid"]["heuristic"]["overall"]
        sh = r["semantic"]["heuristic"]["overall"]
        hl = r["hybrid"]["latency"]
        sl = r["semantic"]["latency"]
        print(
            f"{i:<3}  {q_short:<30}  {hh:>6.1f}  {sh:>6.1f}  "
            f"{_fmt_time(hl):>8}  {_fmt_time(sl):>8}"
        )

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"{'METRIC':<30}  {'HYBRID':>8}  {'SEMANTIC':>8}")
    print("-" * 60)
    print(f"{'Context Precision (RAGAS)':<30}  {_fmt_score(hybrid_cp):>8}  {_fmt_score(semantic_cp):>8}")
    print(f"{'Heuristic Quality (avg)':<30}  {hybrid_avg_heuristic:>8.1f}  {semantic_avg_heuristic:>8.1f}")
    print(f"{'Latency (avg)':<30}  {_fmt_time(hybrid_avg_latency):>8}  {_fmt_time(semantic_avg_latency):>8}")
    print("=" * 60)

    # -------------------------------------------------------------------------
    # Save results
    # -------------------------------------------------------------------------
    output = {
        "timestamp": datetime.now().isoformat(),
        "judge_model": JUDGE_MODEL,
        "n_questions": n,
        "averages": {
            "hybrid": {
                "context_precision": hybrid_cp,
                "heuristic_overall": hybrid_avg_heuristic,
                "latency_s": hybrid_avg_latency,
            },
            "semantic": {
                "context_precision": semantic_cp,
                "heuristic_overall": semantic_avg_heuristic,
                "latency_s": semantic_avg_latency,
            },
        },
        "per_question": per_question,
    }

    with open(RESULTS_FILE, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\nResults saved to: {RESULTS_FILE}")


if __name__ == "__main__":
    main()
