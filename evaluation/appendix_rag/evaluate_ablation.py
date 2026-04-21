"""
evaluate_rag_ablation.py — Retrieved chunks vs random chunks ablation (paper appendix)

Isolates the contribution of retrieval *quality* to answer quality. Two conditions run
over the same 45 synthetic goldens with the Phase 2 winning prompt ("structured"):

  random_chunks — same LLM, same prompt, k=4 chunks sampled uniformly at random
                  from the full corpus. Context is present but irrelevant.
  rag           — same LLM, same prompt, k=4 chunks from hybrid 70/30 retrieval.
                  Context is present and (hopefully) relevant.

Both conditions inject context into the prompt and provide retrieval_context to the
judge, so ALL five metrics apply to both conditions — no "rag-only" split.

Both conditions set use_sensors=False — sensor contribution is measured separately
by evaluate_sensor_awareness.py.

Metric stack (DeepEval-aligned):

  answer_relevancy  Built-in AnswerRelevancyMetric. Does the answer address the
                    user's question regardless of context?

  actionability     Custom G-Eval. Can a hobby keeper act on the answer?

  correctness       Custom G-Eval vs ground_truth. Domain-safety catcher: flags
                    wrong temps, dangerous advice, harmful recommendations.

  faithfulness      Built-in FaithfulnessMetric. Does the answer contradict the
                    provided context? With random chunks, HIGH faithfulness is
                    actually suspicious (model confabulated to match noise).

  contextual_recall Built-in ContextualRecallMetric. Is the information needed
                    for ground_truth actually in the provided chunks? This is the
                    primary retrieval quality signal: random chunks → near zero,
                    retrieved chunks → high.

Output budget for answer generation is raised to 1000 tokens.

Judge: set via SWEEP_JUDGE_MODEL env var.

Usage:
    python evaluation/appendix_rag/evaluate_ablation.py
    python evaluation/appendix_rag/evaluate_ablation.py --n-questions 3   # smoke test
"""

import os
import sys
import json
import random
import time
import argparse
from typing import Dict, List, Optional

os.environ["DEEPEVAL_TELEMETRY_OPT_OUT"] = "YES"

_HERE = os.path.dirname(os.path.abspath(__file__))
_EVAL = os.path.dirname(_HERE)
_ROOT = os.path.dirname(_EVAL)
sys.path.insert(0, _ROOT)
sys.path.insert(0, _EVAL)
sys.path.insert(0, os.path.join(_EVAL, "phase2_prompts"))
sys.path.insert(0, os.path.join(_EVAL, "shared"))

from dotenv import load_dotenv
load_dotenv(dotenv_path=os.path.join(_ROOT, ".env"))

from langchain_openai import ChatOpenAI
from langchain_ollama import OllamaLLM
from deepeval.metrics.g_eval.utils import Rubric
from deepeval.metrics import (
    GEval,
    AnswerRelevancyMetric,
    FaithfulnessMetric,
    ContextualRecallMetric,
)
from deepeval.test_case import LLMTestCase, LLMTestCaseParams

from backend.rag_functions import (
    load_documents, split_documents, build_vector_store,
    build_bm25_retriever, hybrid_search, format_context,
)
from evaluate_variants import OpenRouterJudge, JUDGE_MODEL, PROMPT_VARIANTS
from evaluation_data import TEST_CASES
from eval_config import ACTIONABILITY_CRITERIA, CORRECTNESS_CRITERIA


def load_goldens(path: str) -> List[Dict]:
    with open(path, encoding="utf-8") as f:
        payload = json.load(f)
    goldens = payload.get("goldens", payload)
    rows = []
    for g in goldens:
        rows.append({
            "question":        g["question"],
            "category":        g.get("category", "synthetic"),
            "ground_truth":    g.get("ground_truth", ""),
            "expected_topics": g.get("expected_topics", []),
        })
    return rows


# =============================================================================
# CONFIG
# =============================================================================

PROMPT_TEMPLATE = PROMPT_VARIANTS["structured"]
GEN_MAX_TOKENS  = 1000

ALL_METRICS = [
    "answer_relevancy",
    "actionability",
    "correctness",
    "faithfulness",
    "contextual_recall",
]

CONDITIONS = ["random_chunks", "rag"]


# =============================================================================
# GENERATION
# =============================================================================

def build_generation_llm(model: str, max_tokens: int = GEN_MAX_TOKENS):
    if model.startswith("openrouter/"):
        return ChatOpenAI(
            model=model.removeprefix("openrouter/"),
            base_url="https://openrouter.ai/api/v1",
            api_key=os.getenv("OPENROUTER_API_KEY"),
            temperature=0.2,
            max_tokens=max_tokens,
        )
    return OllamaLLM(
        model=model,
        temperature=0.2,
        num_predict=max_tokens,
        repeat_penalty=1.1,
        top_k=40,
        top_p=0.9,
    )


def generate_answer(llm, prompt: str) -> str:
    response = llm.invoke(prompt)
    text = response.content if hasattr(response, "content") else response
    import re
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


# =============================================================================
# G-EVAL RUBRICS
# =============================================================================

_ACTION_RUBRIC = [
    Rubric(score_range=(0, 3), expected_outcome="Not actionable — vague generalities, nothing to act on, repeats the question, or just restates the problem."),
    Rubric(score_range=(4, 5), expected_outcome="Partially actionable — contains some useful information but missing key specifics, overly hedged, or leaves important gaps the user must fill themselves."),
    Rubric(score_range=(6, 7), expected_outcome="Helpful — answers the question clearly with at least one concrete action or specific fact; a hobby keeper can act on it, though minor gaps or hedging remain."),
    Rubric(score_range=(8, 10), expected_outcome="Precisely actionable — directly targeted to the specific question, covers all key steps or facts with appropriate detail, calibrated to the user's actual situation, no significant gaps or unnecessary hedging. Reserve 9-10 for answers that are both complete AND notably well-tailored."),
]

_CORRECT_RUBRIC = [
    Rubric(score_range=(0, 3), expected_outcome="Incorrect or harmful — contains wrong facts that could mislead a beginner, dangerous advice, or actively contradicts the reference answer or established poultry practice."),
    Rubric(score_range=(4, 5), expected_outcome="Partially correct — the core claim is broadly right but misses important details present in the reference answer, contains a meaningful inaccuracy, or is so hedged it becomes unreliable."),
    Rubric(score_range=(6, 7), expected_outcome="Correct — accurate and consistent with standard practice and the reference answer on main points; minor omissions or imprecisions acceptable, but nothing that would mislead the user."),
    Rubric(score_range=(8, 10), expected_outcome="Fully correct — accurate, complete, aligns closely with the reference answer on all key points. Calibrated detail, no significant omissions or inaccuracies. Reserve 9-10 for answers that match both the facts AND the practical emphasis of the reference."),
]


# =============================================================================
# METRIC BUILDERS
# =============================================================================

def build_metrics(judge: OpenRouterJudge) -> Dict[str, object]:
    return {
        "answer_relevancy": AnswerRelevancyMetric(
            model=judge, include_reason=False, async_mode=False,
        ),
        "actionability": GEval(
            name="Actionability",
            criteria=ACTIONABILITY_CRITERIA,
            evaluation_params=[
                LLMTestCaseParams.INPUT,
                LLMTestCaseParams.ACTUAL_OUTPUT,
            ],
            model=judge,
            rubric=_ACTION_RUBRIC,
            async_mode=False,
        ),
        "correctness": GEval(
            name="Correctness",
            criteria=CORRECTNESS_CRITERIA,
            evaluation_params=[
                LLMTestCaseParams.INPUT,
                LLMTestCaseParams.ACTUAL_OUTPUT,
                LLMTestCaseParams.EXPECTED_OUTPUT,
            ],
            model=judge,
            rubric=_CORRECT_RUBRIC,
            async_mode=False,
        ),
        "faithfulness": FaithfulnessMetric(
            model=judge, include_reason=False, async_mode=False,
        ),
        "contextual_recall": ContextualRecallMetric(
            model=judge, include_reason=False, async_mode=False,
        ),
    }


# =============================================================================
# RETRY
# =============================================================================

def _measure_with_retry(metric, test_case: LLMTestCase, max_attempts: int = 5) -> float:
    last_err = None
    for attempt in range(1, max_attempts + 1):
        try:
            metric.measure(test_case)
            score = float(metric.score)
            if score < 0.0 or score > 1.0:
                print(f"    [out-of-range score {score:.3f} → -1 sentinel]")
                return -1.0
            return score
        except Exception as e:
            last_err = e
            if attempt < max_attempts:
                print(f"    [judge attempt {attempt}/{max_attempts} failed: {type(e).__name__}] retrying...")
                time.sleep(1.5 * attempt)
    print(f"    [judge failure x{max_attempts}, returning -1 sentinel] {type(last_err).__name__}: {last_err}")
    return -1.0


def _score_test_case(
    test_case: LLMTestCase,
    metrics: Dict[str, object],
) -> Dict[str, float]:
    scores: Dict[str, float] = {}
    for name in ALL_METRICS:
        scores[name] = _measure_with_retry(metrics[name], test_case)
    return scores


# =============================================================================
# RUNNER
# =============================================================================

def _checkpoint_path(results_dir: str, condition: str) -> str:
    return os.path.join(results_dir, f"rag_ablation_{condition}_checkpoint.jsonl")


def _load_checkpoint(path: str) -> Dict[str, Dict]:
    done = {}
    if not os.path.exists(path):
        return done
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                row = json.loads(line)
                done[row["question"]] = row
    return done


def _append_checkpoint(path: str, row: Dict):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def run_condition(
    condition: str,
    questions: List[Dict],
    all_chunks: list,
    vectordb,
    bm25_retriever,
    llm,
    metrics: Dict[str, object],
    k: int = 4,
    weights: list = None,
    results_dir: str = None,
    seed: int = 42,
) -> List[Dict]:
    """Run one condition over all questions. Both conditions inject context and
    provide retrieval_context to the judge, so all five metrics apply to both."""
    if weights is None:
        weights = [0.7, 0.3]

    rng = random.Random(seed)

    ckpt_path = _checkpoint_path(results_dir, condition) if results_dir else None
    done = _load_checkpoint(ckpt_path) if ckpt_path else {}
    if done:
        print(f"  [checkpoint] resuming — {len(done)} questions already done")

    results = list(done.values())

    for i, test in enumerate(questions, 1):
        question    = test["question"]
        category    = test["category"]
        ground_truth = test.get("ground_truth", "")

        if question in done:
            print(f"  [{i}/{len(questions)}] {category}: skipped (checkpoint)")
            continue

        if condition == "rag":
            docs = hybrid_search(vectordb, bm25_retriever, question, k=k, weights=weights)
            sources = [d.metadata.get("source", "?") for d in docs]
        elif condition == "random_chunks":
            # Sample k chunks uniformly at random from the full corpus.
            # No exclusion of "correct" chunks — that would be peeking. Pure random draw.
            docs = rng.sample(all_chunks, min(k, len(all_chunks)))
            sources = [d.metadata.get("source", "?") for d in docs]
        else:
            raise ValueError(f"Unknown condition: {condition}")

        context = format_context(docs)
        retrieval_context = [d.page_content for d in docs]

        prompt = PROMPT_TEMPLATE.format(context=context, query=question)
        t0 = time.time()
        answer = generate_answer(llm, prompt)
        elapsed = time.time() - t0

        test_case = LLMTestCase(
            input=question,
            actual_output=answer,
            expected_output=ground_truth,
            retrieval_context=retrieval_context,
        )

        scores = _score_test_case(test_case, metrics)

        row = {
            "condition":    condition,
            "question":     question,
            "category":     category,
            "ground_truth": ground_truth,
            "answer":       answer,
            "sources":      sources,
            "time":         round(elapsed, 2),
            **scores,
        }
        results.append(row)
        if ckpt_path:
            _append_checkpoint(ckpt_path, row)

        parts = [
            f"ar={scores['answer_relevancy']:.2f}",
            f"act={scores['actionability']:.2f}",
            f"cor={scores['correctness']:.2f}",
            f"faith={scores['faithfulness']:.2f}",
            f"crec={scores['contextual_recall']:.2f}",
        ]
        print(f"  [{i}/{len(questions)}] {category}: " + "  ".join(parts) + f"  {elapsed:.1f}s")

    return results


# =============================================================================
# SUMMARY
# =============================================================================

def _mean(values: List[Optional[float]]) -> float:
    vals = [v for v in values if v is not None and v >= 0]
    return sum(vals) / len(vals) if vals else float("nan")


def print_summary(all_results: Dict[str, List[Dict]]):
    print("\n" + "=" * 96)
    print("SUMMARY  (means, 0–1 scale)")
    print("=" * 96)
    header = f"{'Condition':<15} " + "  ".join(f"{m[:10]:>10}" for m in ALL_METRICS)
    print(header)
    print("-" * 96)
    for condition in CONDITIONS:
        results = all_results.get(condition, [])
        cells = [f"{condition:<15}"]
        for name in ALL_METRICS:
            m = _mean([r.get(name) for r in results])
            cells.append(f"{m:>10.4f}" if m == m else f"{'—':>10}")
        print(" ".join(cells))


# =============================================================================
# MAIN
# =============================================================================

def run_evaluation(
    n_questions: int = None,
    k: int = 4,
    weights: list = None,
    llm_model: str = None,
    chunk_size: int = 1000,
    goldens_file: str = None,
    seed: int = 42,
):
    if weights is None:
        weights = [0.7, 0.3]
    if llm_model is None:
        llm_model = os.getenv("OLLAMA_MODEL", "smollm2:1.7b")

    KB_PATH     = os.path.join(_ROOT, "test_docs")
    CHROMA_DIR  = os.path.join(_ROOT, "chroma_db")
    RESULTS_DIR = os.path.join(_EVAL, "results")
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("=" * 96)
    print("RETRIEVED vs RANDOM CHUNKS ABLATION  (DeepEval-aligned metric stack)")
    print(f"  LLM:         {llm_model}  (max_tokens={GEN_MAX_TOKENS})")
    print(f"  Judge:       {JUDGE_MODEL}")
    print(f"  Retrieval:   hybrid 70/30, k={k}, chunk={chunk_size}")
    print(f"  Prompt:      structured (Phase 2 winner)")
    print(f"  Metrics:     {', '.join(ALL_METRICS)}")
    print(f"  Conditions:  {CONDITIONS}")
    print(f"  Random seed: {seed}")
    print("=" * 96)

    print("\nSetting up AI judge...")
    judge = OpenRouterJudge(model_name=JUDGE_MODEL)
    metrics = build_metrics(judge)

    print(f"Building generation LLM (max_tokens={GEN_MAX_TOKENS})...")
    llm = build_generation_llm(llm_model, max_tokens=GEN_MAX_TOKENS)

    print("Building RAG pipeline...")
    docs      = load_documents(KB_PATH)
    chunks    = split_documents(docs, chunk_size=chunk_size)
    vectordb  = build_vector_store(chunks, persist_dir=CHROMA_DIR, folder_path=KB_PATH)
    bm25      = build_bm25_retriever(chunks)
    print(f"RAG pipeline ready. Corpus: {len(chunks)} chunks\n")

    if goldens_file:
        questions = load_goldens(goldens_file)
        print(f"  Source:      {goldens_file}  ({len(questions)} goldens)")
    else:
        questions = TEST_CASES
    questions = questions[:n_questions] if n_questions else questions

    all_results: Dict[str, List[Dict]] = {}

    for condition in CONDITIONS:
        print(f"\n{'-' * 60}")
        print(f"Condition: {condition.upper()}")
        print(f"{'-' * 60}")
        all_results[condition] = run_condition(
            condition=condition,
            questions=questions,
            all_chunks=chunks,
            vectordb=vectordb,
            bm25_retriever=bm25,
            llm=llm,
            metrics=metrics,
            k=k,
            weights=weights,
            results_dir=RESULTS_DIR,
            seed=seed,
        )

    print_summary(all_results)

    out_path = os.path.join(RESULTS_DIR, "rag_ablation_results.json")
    payload = {
        "experiment":            "retrieved_vs_random_chunks",
        "judge_model":           JUDGE_MODEL,
        "llm_model":             llm_model,
        "generation_max_tokens": GEN_MAX_TOKENS,
        "retrieval":             {"k": k, "weights": weights, "chunk_size": chunk_size},
        "random_seed":           seed,
        "prompt_template":       "structured",
        "question_source":       goldens_file or "evaluation_data.TEST_CASES",
        "n_questions":           len(questions),
        "metrics":               ALL_METRICS,
        "conditions":            all_results,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to: {out_path}")

    for condition in all_results:
        ckpt = _checkpoint_path(RESULTS_DIR, condition)
        if os.path.exists(ckpt):
            os.remove(ckpt)
            print(f"Checkpoint removed: {ckpt}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Retrieved vs random chunks ablation")
    parser.add_argument("--n-questions", type=int, default=None,
                        help="Limit to first N questions (smoke test)")
    parser.add_argument("--k", type=int, default=4)
    parser.add_argument("--chunk-size", type=int, default=1000)
    parser.add_argument("--model", type=str, default=None,
                        help="LLM model override (e.g. openrouter/mistralai/ministral-8b)")
    parser.add_argument("--goldens-file", type=str, default=None,
                        help="Path to goldens JSON; overrides evaluation_data.TEST_CASES")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for chunk sampling")
    args = parser.parse_args()

    run_evaluation(
        n_questions=args.n_questions,
        k=args.k,
        llm_model=args.model,
        chunk_size=args.chunk_size,
        goldens_file=args.goldens_file,
        seed=args.seed,
    )
