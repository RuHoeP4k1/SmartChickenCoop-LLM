# evaluation/evaluate_ragas.py
"""
RAG vs NO-RAG evaluation using RAGAS semantic metrics.

Compares:
  RAG    = hybrid retrieval (60% semantic + 40% BM25) + Qwen LLM
  NO-RAG = Qwen LLM with no document retrieval

Metrics: faithfulness, answer_relevancy, context_precision, context_recall

NOTE: Scores will be imprecise with Qwen 1.5b as the judge model.
If you see many NaN or unstable values, switch judge to qwen2.5:7b
(change JUDGE_MODEL below). The generator model stays at 1.5b.

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
from ragas.metrics import Faithfulness, AnswerRelevancy, ContextPrecision, ContextRecall
from ragas.run_config import RunConfig

from rag_functions import (
    load_documents, split_documents, build_vector_store,
    build_bm25_retriever, answer_query,
)
from evaluate_rag import get_norag_answer
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
RESULTS_FILE = os.path.join(RESULTS_DIR, "ragas_results.json")


def _avg_from_eval(eval_result, metric_names: list) -> dict:
    """Extract per-metric averages from a ragas 0.4.x Results object.

    In ragas 0.4.x Results is a Pydantic model — dict() gives model fields
    (scores, dataset), NOT metric averages. Use subscript instead:
    eval_result["metric_name"] returns a list[float] of per-sample scores.
    """
    out = {}
    for m in metric_names:
        try:
            vals = eval_result[m]  # list[float | None]
            valid = [v for v in vals if v is not None and v == v]
            out[m] = sum(valid) / len(valid) if valid else float("nan")
        except Exception:
            out[m] = float("nan")
    return out


def _fmt(val) -> str:
    if val != val:  # NaN check
        return "  N/A "
    return f"{val:.4f}"


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
    # Embeddings always come from local Ollama (used by AnswerRelevancy)
    judge_embeddings = LangchainEmbeddingsWrapper(
        OllamaEmbeddings(model=os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text-v2-moe"))
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
    # Collect answers for both conditions
    # -------------------------------------------------------------------------
    rag_samples = []
    norag_samples = []
    raw_results = []

    for i, tc in enumerate(TEST_CASES, 1):
        question = tc["question"]
        ground_truth = tc["ground_truth"]

        print(f"[{i}/{len(TEST_CASES)}] {question}")

        # RAG: retrieve + generate (contexts taken from same call to avoid MMR stochasticity)
        t0 = time.time()
        rag_result = answer_query(
            question, vectordb, bm25, use_sensors=False, use_hybrid=True,
        )
        contexts = [doc.page_content for doc in rag_result["documents"]]
        rag_time = time.time() - t0

        # NO-RAG: generate only
        t0 = time.time()
        norag_result = get_norag_answer(question)
        norag_time = time.time() - t0

        rag_samples.append(
            SingleTurnSample(
                user_input=question,
                response=rag_result["answer"],
                retrieved_contexts=contexts,
                reference=ground_truth,
            )
        )
        norag_samples.append(
            SingleTurnSample(
                user_input=question,
                response=norag_result["answer"],
                retrieved_contexts=[],  # no retrieval for NO-RAG
                reference=ground_truth,
            )
        )

        raw_results.append({
            "question": question,
            "category": tc["category"],
            "rag_answer": rag_result["answer"],
            "norag_answer": norag_result["answer"],
            "rag_sources": rag_result["sources"],
            "rag_contexts": contexts,
            "rag_time": round(rag_time, 3),
            "norag_time": round(norag_time, 3),
            "ground_truth": ground_truth,
        })

    # -------------------------------------------------------------------------
    # Run RAGAS evaluation
    # -------------------------------------------------------------------------
    # ragas 0.4.x requires llm/embeddings passed to metric constructors directly
    metrics = [
        Faithfulness(llm=judge_llm),
        AnswerRelevancy(llm=judge_llm, embeddings=judge_embeddings),
        ContextPrecision(llm=judge_llm),
        ContextRecall(llm=judge_llm),
    ]
    metric_names = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]

    # run_config: max_workers=1 serialises calls so local Ollama isn't overwhelmed;
    # timeout=600 gives each LLM call up to 10 minutes (qwen3:4b is slower than 1.5b).
    _run_cfg = RunConfig(timeout=600, max_workers=1)

    print("\nRunning RAGAS on RAG condition (this may take a while)...")
    rag_dataset = EvaluationDataset(samples=rag_samples)
    rag_eval = evaluate(
        rag_dataset, metrics=metrics, llm=judge_llm, embeddings=judge_embeddings,
        run_config=_run_cfg, raise_exceptions=False,
    )

    print("Running RAGAS on NO-RAG condition...")
    norag_dataset = EvaluationDataset(samples=norag_samples)
    norag_eval = evaluate(
        norag_dataset, metrics=metrics, llm=judge_llm, embeddings=judge_embeddings,
        run_config=_run_cfg, raise_exceptions=False,
    )

    # -------------------------------------------------------------------------
    # Extract scores
    # -------------------------------------------------------------------------
    rag_scores = _avg_from_eval(rag_eval, metric_names)
    norag_scores = _avg_from_eval(norag_eval, metric_names)

    # -------------------------------------------------------------------------
    # Print side-by-side comparison table
    # -------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("RAGAS RESULTS — RAG vs NO-RAG")
    print("=" * 60)
    print(f"{'METRIC':<25}  {'RAG':>7}  {'NO-RAG':>7}  {'DIFF':>7}")
    print("-" * 60)
    for m in metric_names:
        rag_v = rag_scores[m]
        norag_v = norag_scores[m]
        if rag_v == rag_v and norag_v == norag_v:  # both not NaN
            diff = f"{rag_v - norag_v:+.4f}"
        else:
            diff = "   N/A"
        print(f"{m:<25}  {_fmt(rag_v):>7}  {_fmt(norag_v):>7}  {diff:>7}")
    print("=" * 60)

    # -------------------------------------------------------------------------
    # Save results to JSON
    # -------------------------------------------------------------------------
    output = {
        "timestamp": datetime.now().isoformat(),
        "judge_model": JUDGE_MODEL,
        "n_questions": len(TEST_CASES),
        "averages": {
            "rag": rag_scores,
            "norag": norag_scores,
        },
        "per_question": raw_results,
        "raw_rag_eval": str(rag_eval),
        "raw_norag_eval": str(norag_eval),
    }

    with open(RESULTS_FILE, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\nResults saved to: {RESULTS_FILE}")


if __name__ == "__main__":
    main()
