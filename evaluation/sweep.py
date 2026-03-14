# evaluation/sweep.py
"""
RAG hyperparameter sweep — D-Optimal design runner.

Runs each config in the D-Optimal design matrix, scores every answer with
DeepEval G-Eval (actionability + correctness), and saves results with
full checkpoint/resume support.

Usage:
    python evaluation/sweep.py                    # full sweep (18 runs, main effects)
    python evaluation/sweep.py --dry-run          # print design table only
    python evaluation/sweep.py --n-questions 3    # smoke test (3 questions per config)
    python evaluation/sweep.py --n-runs 6         # smoke test (6 configs only)

Prerequisites:
    pip install pyDOE3 langchain-openai deepeval langchain-anthropic
    OPENROUTER_API_KEY in .env (free tier: llama models + Gemini judge)
    OPENAI_API_KEY in .env (for text-embedding-3-small only)
"""

import os
import sys
import csv
import json
import time
import random
import argparse
from datetime import datetime
from pathlib import Path

os.environ["DEEPEVAL_TELEMETRY_OPT_OUT"] = "YES"

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _ROOT)
sys.path.insert(0, _HERE)

from dotenv import load_dotenv
load_dotenv(dotenv_path=os.path.join(_ROOT, ".env"))

from langchain_openai import ChatOpenAI
try:
    from langchain_anthropic import ChatAnthropic
    _ANTHROPIC_AVAILABLE = True
except ImportError:
    _ANTHROPIC_AVAILABLE = False
from deepeval.models.base_model import DeepEvalBaseLLM
from deepeval.metrics.g_eval.utils import Rubric
from deepeval.errors import MissingTestCaseParamsError
from pydantic import ValidationError
from deepeval.metrics import (
    GEval,
    FaithfulnessMetric,
    AnswerRelevancyMetric,
    ContextualRecallMetric,
)
from deepeval.test_case import LLMTestCase, LLMTestCaseParams

from backend.rag_functions import (
    load_documents, split_documents, build_vector_store,
    build_bm25_retriever, answer_query,
)
from evaluation_data import TEST_CASES
from eval_config import ACTIONABILITY_CRITERIA, CORRECTNESS_CRITERIA
from sweep_config import (
    build_design, config_id, chroma_dir, use_hybrid_flag,
    save_design_md, FACTOR_KEYS, EMBED_MODEL,
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
KB_PATH     = os.path.join(_ROOT, "test_docs")
RESULTS_DIR = os.path.join(_HERE, "results")
TIMESTAMP   = datetime.now().strftime("%Y%m%d_%H%M%S")
RESULTS_FILE     = os.path.join(RESULTS_DIR, f"sweep_{TIMESTAMP}.json")
RESULTS_CSV      = os.path.join(RESULTS_DIR, f"sweep_{TIMESTAMP}.csv")
CHECKPOINT_FILE  = os.path.join(RESULTS_DIR, "sweep_checkpoint.json")
DESIGN_MD        = os.path.join(RESULTS_DIR, "doe_design.md")

# ---------------------------------------------------------------------------
# Judge models
# ---------------------------------------------------------------------------
JUDGE_MODEL = os.getenv("SWEEP_JUDGE_MODEL", "google/gemini-2.0-flash-exp:free")

_judge_token_usage = {"input": 0, "output": 0}


def _track_usage(response):
    usage = getattr(response, "usage_metadata", None)
    if isinstance(usage, dict):
        _judge_token_usage["input"]  += usage.get("input_tokens",  0)
        _judge_token_usage["output"] += usage.get("output_tokens", 0)
    elif usage is not None:
        _judge_token_usage["input"]  += getattr(usage, "input_tokens",  0)
        _judge_token_usage["output"] += getattr(usage, "output_tokens", 0)


def print_token_usage(label: str):
    inp, out = _judge_token_usage["input"], _judge_token_usage["output"]
    if inp + out == 0:
        return
    print(f"\nJudge token usage ({label}): {inp:,} in / {out:,} out")
    print(f"  Check OpenRouter or Anthropic dashboard for exact cost.")


class OpenRouterJudge(DeepEvalBaseLLM):
    """DeepEval-compatible judge via OpenRouter (tracks token usage)."""

    def __init__(self, model_name: str = JUDGE_MODEL):
        self.model_name = model_name
        self.model = ChatOpenAI(
            model=model_name,
            base_url="https://openrouter.ai/api/v1",
            api_key=os.getenv("OPENROUTER_API_KEY"),
            temperature=0,
            max_tokens=2048,  # covers both evaluation_steps generation and score JSON
        )

    def load_model(self):
        return self.model

    def generate(self, prompt: str) -> str:
        response = self.model.invoke(prompt)
        _track_usage(response)
        return response.content if hasattr(response, "content") else str(response)

    async def a_generate(self, prompt: str) -> str:
        return self.generate(prompt)

    def get_model_name(self) -> str:
        return f"OpenRouter/{self.model_name}"


# ---------------------------------------------------------------------------
# Claude Haiku judge (Anthropic API)
# ---------------------------------------------------------------------------

class ClaudeHaikuJudge(DeepEvalBaseLLM):
    """DeepEval-compatible judge using Claude Haiku via Anthropic API."""

    def __init__(self, model_name: str = "claude-haiku-4-5-20251001"):
        self.model_name = model_name
        self.model = ChatAnthropic(
            model=model_name,
            api_key=os.getenv("ANTHROPIC_API_KEY"),
            temperature=0,
        )

    def load_model(self):
        return self.model

    def generate(self, prompt: str) -> str:
        response = self.model.invoke(prompt)
        _track_usage(response)
        return response.content if hasattr(response, "content") else str(response)

    async def a_generate(self, prompt: str) -> str:
        return self.generate(prompt)

    def get_model_name(self) -> str:
        return f"Anthropic/{self.model_name}"


# ---------------------------------------------------------------------------
# Rate-limit backoff (OpenRouter free tier)
# ---------------------------------------------------------------------------

def call_with_backoff(fn, max_retries: int = 8):
    """Retry fn() with exponential backoff on 429 rate-limit errors."""
    for attempt in range(max_retries):
        try:
            return fn()
        except Exception as e:
            if "429" in str(e) or "rate" in str(e).lower():
                wait = (3 ** attempt) + random.uniform(1, 3)
                print(f"  [rate limit] waiting {wait:.1f}s (attempt {attempt + 1}/{max_retries})")
                time.sleep(wait)
            else:
                raise
    raise RuntimeError(f"Failed after {max_retries} retries")


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def load_checkpoint() -> list:
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE, encoding="utf-8") as f:
            return json.load(f)
    return []


def save_checkpoint(results: list) -> None:
    Path(RESULTS_DIR).mkdir(parents=True, exist_ok=True)
    with open(CHECKPOINT_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)


def already_done(cfg: dict, done_results: list) -> bool:
    cid = config_id(cfg)
    return any(config_id(r["config"]) == cid for r in done_results)


# ---------------------------------------------------------------------------
# Per-config pipeline cache
# (avoids re-embedding the same (embed, chunk) pair across configs)
# ---------------------------------------------------------------------------

_pipeline_cache: dict = {}   # key: (embed_slug, chunk_size) → (vectordb, bm25)


def get_pipeline(cfg: dict):
    """Return (vectordb, bm25) for the given config, building only if needed."""
    key = ("nomic", cfg["chunk_size"])
    if key not in _pipeline_cache:
        print(f"  [pipeline] building for embed=nomic, chunk={key[1]}...")
        docs   = load_documents(KB_PATH)
        chunks = split_documents(docs, chunk_size=cfg["chunk_size"])
        vdb    = build_vector_store(
            chunks,
            persist_dir=chroma_dir(cfg),
            folder_path=KB_PATH,
            embedding_model=EMBED_MODEL,
        )
        bm25   = build_bm25_retriever(chunks)
        _pipeline_cache[key] = (vdb, bm25)
    else:
        print(f"  [pipeline] reusing cached embed=nomic, chunk={key[1]}")
    return _pipeline_cache[key]


# ---------------------------------------------------------------------------
# Score one config
# ---------------------------------------------------------------------------

def run_config(
    cfg: dict,
    test_cases: list,
    actionability_metric: GEval,
    correctness_metric: GEval,
    faithfulness_metric,
    answer_relevancy_metric,
    ctx_recall_metric,
) -> dict:
    """Run all test questions for one config and return scored results."""
    vectordb, bm25 = get_pipeline(cfg)
    hybrid = use_hybrid_flag(cfg)

    per_question = []
    for tc in test_cases:
        question = tc["question"]

        t0 = time.time()
        result = call_with_backoff(lambda: answer_query(
            question,
            vectordb,
            bm25,
            use_sensors=False,
            use_hybrid=hybrid,
            k=cfg["k"],
            search_type="similarity",
            weights=cfg["weights"] if hybrid else None,
            llm_model=cfg["llm_model"],
        ))
        latency = round(time.time() - t0, 3)

        answer   = result["answer"]
        docs     = result["documents"]
        gt       = tc.get("ground_truth", "")
        ret_ctx  = [doc.page_content for doc in docs]

        if not answer:
            print(f"  ⚠ Empty answer for '{question}' (model returned only thinking tokens?) — skipping all metrics")
            per_question.append({
                "question":             question,
                "category":             tc.get("category", "unknown"),
                "answer":               "",
                "actionability":        -1.0,
                "correctness":          -1.0,
                "faithfulness":         -1.0,
                "answer_relevancy":     -1.0,
                "contextual_recall":    -1.0,
                "latency":              latency,
            })
            continue

        test_case = LLMTestCase(
            input=question,
            actual_output=answer,
            expected_output=gt,
            retrieval_context=ret_ctx,
        )

        action_score    = call_with_backoff(lambda: _measure(actionability_metric, test_case))
        correct_score   = call_with_backoff(lambda: _measure(correctness_metric, test_case))
        faith_score     = call_with_backoff(lambda: _measure(faithfulness_metric, test_case))
        ans_rel_score   = call_with_backoff(lambda: _measure(answer_relevancy_metric, test_case))
        ctx_rec_score   = call_with_backoff(lambda: _measure(ctx_recall_metric, test_case))

        per_question.append({
            "question":             question,
            "category":             tc.get("category", "unknown"),
            "answer":               answer,
            "actionability":        action_score,     # 0-1 (DeepEval normalises GEval internally)
            "correctness":          correct_score,    # 0-1
            "faithfulness":         faith_score,      # 0-1
            "answer_relevancy":     ans_rel_score,    # 0-1
            "contextual_recall":    ctx_rec_score,    # 0-1
            "latency":              latency,
        })

    n = len(per_question)

    def _avg(key):
        vals = [q[key] for q in per_question if isinstance(q[key], (int, float)) and q[key] >= 0]
        if not vals:
            return -1.0
        return sum(vals) / len(vals)

    def _fail_rate(key):
        fails = sum(1 for q in per_question if q.get(key, 0) < 0)
        return round(fails / n, 3) if n else 0.0

    avg_action  = _avg("actionability")
    avg_correct = _avg("correctness")
    # combined: only use metrics that actually succeeded
    combined_vals = [v for v in [avg_action, avg_correct] if v >= 0]
    avg_combined = sum(combined_vals) / len(combined_vals) if combined_vals else -1.0

    fail_rates = {m: _fail_rate(m) for m in [
        "actionability", "correctness", "faithfulness",
        "answer_relevancy", "contextual_recall",
    ]}
    # Warn if any metric fails on more than 20% of questions
    for m, rate in fail_rates.items():
        if rate > 0.2:
            print(f"  WARNING: {m} failed on {rate*100:.0f}% of questions — results unreliable")

    return {
        "config": cfg,
        "avg_actionability":        round(avg_action, 4),
        "avg_correctness":          round(avg_correct, 4),
        "avg_combined":             round(avg_combined, 4),
        "avg_faithfulness":         round(_avg("faithfulness"), 4),
        "avg_answer_relevancy":     round(_avg("answer_relevancy"), 4),
        "avg_contextual_recall":    round(_avg("contextual_recall"), 4),
        "metric_fail_rates":        fail_rates,
        "per_question": per_question,
    }


def _measure(metric, test_case: LLMTestCase) -> float:
    """Measure a DeepEval metric and return its score. Returns -1 on failure."""
    for attempt in range(2):  # 1 retry on JSON/key errors
        try:
            metric.measure(test_case)
            score = metric.score
            if not (0.0 <= score <= 1.0):
                print(f"  ⚠ {metric.__class__.__name__} returned {score} (out of 0-1 range) — treating as failure")
                return -1.0
            return score
        except MissingTestCaseParamsError as e:
            print(f"  ⚠ Empty field in {metric.__class__.__name__} — skipping (score=-1): {e}")
            return -1.0
        except (KeyError, ValidationError) as e:
            if attempt == 0:
                print(f"  ⚠ Malformed judge output in {metric.__class__.__name__} — retrying once…")
                time.sleep(3)
                continue
            print(f"  ⚠ Malformed judge output in {metric.__class__.__name__} after retry — skipping (score=-1)")
            return -1.0
        except (ValueError, json.JSONDecodeError) as e:
            if "json" in str(e).lower() or isinstance(e, json.JSONDecodeError):
                if attempt == 0:
                    print(f"  ⚠ JSON parse error in {metric.__class__.__name__} — retrying once…")
                    time.sleep(3)
                    continue
                print(f"  ⚠ JSON parse error in {metric.__class__.__name__} after retry — skipping (score=-1)")
                return -1.0
            raise
    return -1.0  # unreachable but satisfies type checker


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

def print_ranking_table(results: list) -> None:
    sorted_r = sorted(results, key=lambda r: r["avg_combined"], reverse=True)
    print("\n" + "=" * 80)
    print("SWEEP RESULTS — TOP CONFIGURATIONS")
    print("=" * 80)
    header = (f"{'Rank':<5} {'LLM':<15} {'Chunk':<6} {'k':<3} {'Weights':<10} "
              f"{'Action':>7} {'Correct':>8} {'Faithful':>9} {'Combined':>9}")
    print(header)
    print("-" * 80)
    for rank, r in enumerate(sorted_r[:15], 1):
        cfg     = r["config"]
        llm     = _short(cfg["llm_model"], 14)
        chunk   = str(cfg["chunk_size"])
        k       = str(cfg["k"])
        w       = cfg["weights"]
        weights = "pure-sem" if w == [0.0, 1.0] else "70/30"
        print(
            f"{rank:<5} {llm:<15} {chunk:<6} {k:<3} {weights:<10} "
            f"{r['avg_actionability']:>7.4f} {r['avg_correctness']:>8.4f} "
            f"{r.get('avg_faithfulness', 0):>9.4f} {r['avg_combined']:>9.4f}"
        )
    print("=" * 80)


def _short(s: str, n: int) -> str:
    if "qwen3-8b"  in s: return "qwen3-8b"
    if "ministral" in s: return "ministral-14b"
    if "mistral-small" in s: return "mistral-small-24b"
    if "3.3-70b"   in s: return "llama-3.3-70b"
    return s[:n]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="RAG hyperparameter sweep")
    parser.add_argument("--n-runs",      type=int, default=None,  help="Number of runs (default: auto from LEVELS)")
    parser.add_argument("--n-questions", type=int, default=None,  help="Limit questions per config (smoke test)")
    parser.add_argument("--dry-run",     action="store_true",     help="Print design table and exit")
    parser.add_argument("--fresh",       action="store_true",     help="Ignore checkpoint, start from scratch")
    parser.add_argument("--judge",       choices=["haiku", "openrouter"], default=None,
                        help="Force judge: 'haiku' (Anthropic) or 'openrouter' (uses SWEEP_JUDGE_MODEL)")
    args = parser.parse_args()

    Path(RESULTS_DIR).mkdir(parents=True, exist_ok=True)

    # Build design
    from sweep_config import LEVELS as _LEVELS
    import itertools as _it
    full_size = 1
    for lv in _LEVELS:
        full_size *= lv
    n_runs = args.n_runs  # None → full factorial
    print(f"Generating design ({full_size} runs — full factorial)...")
    configs = build_design(n_runs=n_runs)
    save_design_md(configs, DESIGN_MD)
    print(f"Design matrix saved -> {DESIGN_MD}")

    if args.dry_run:
        print(f"\nDry run complete. {len(configs)} configurations planned.")
        print("\n--- Design size reference ---")
        print(f"  Current  LEVELS={_LEVELS}: {full_size} runs  <- active")
        _l24 = [3, 2, 2, 2]
        _s24 = 1
        for lv in _l24: _s24 *= lv
        print(f"  Round 1  LEVELS={_l24}: {_s24} runs  (set in sweep_config.py)")
        _l54 = [3, 3, 3, 2]
        _s54 = 1
        for lv in _l54: _s54 *= lv
        print(f"  Extended LEVELS={_l54}: {_s54} runs  (optional, +800 chunk + k=3)")
        return

    # Truncate test cases for smoke testing
    test_cases = TEST_CASES[:args.n_questions] if args.n_questions else TEST_CASES
    print(f"\nTest questions: {len(test_cases)} | Configs: {len(configs)} | Judge: {JUDGE_MODEL}")

    # Load checkpoint
    results = [] if args.fresh else load_checkpoint()
    if results:
        print(f"Resuming from checkpoint: {len(results)}/{len(configs)} configs already done.")

    # Judge setup
    anthropic_key = os.getenv("ANTHROPIC_API_KEY")
    use_haiku = (args.judge == "haiku") or (args.judge is None and anthropic_key and _ANTHROPIC_AVAILABLE)
    use_openrouter = (args.judge == "openrouter") or not use_haiku
    if use_haiku and not use_openrouter:
        judge = ClaudeHaikuJudge()
        judge_label = "Claude Haiku (Anthropic)"
    else:
        judge = OpenRouterJudge(model_name=JUDGE_MODEL)
        judge_label = JUDGE_MODEL
    print(f"Judge: {judge_label}")

    # Custom G-Eval metrics (0–10 rubric, normalized to 0–1 by DeepEval)
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

    # Standard DeepEval RAG metrics (0–1 scale)
    faithfulness_metric      = FaithfulnessMetric(model=judge, include_reason=False)
    answer_relevancy_metric  = AnswerRelevancyMetric(model=judge, include_reason=False)
    ctx_recall_metric        = ContextualRecallMetric(model=judge, include_reason=False)

    # Run sweep
    for i, cfg in enumerate(configs, 1):
        if already_done(cfg, results):
            print(f"[{i}/{len(configs)}] SKIP (checkpoint)")
            continue

        llm_label = _short(cfg["llm_model"], 20)
        w = cfg["weights"]
        w_label = "pure-sem" if w == [0.0, 1.0] else "70/30"
        print(f"\n[{i}/{len(configs)}] llm={llm_label} chunk={cfg['chunk_size']} k={cfg['k']} weights={w_label}")

        result = run_config(
            cfg, test_cases,
            actionability_metric, correctness_metric,
            faithfulness_metric, answer_relevancy_metric,
            ctx_recall_metric,
        )
        results.append(result)
        save_checkpoint(results)

        print(f"  → action={result['avg_actionability']:.4f}  correct={result['avg_correctness']:.4f}  "
              f"faithful={result['avg_faithfulness']:.4f}  combined={result['avg_combined']:.4f}")

    # Final save
    with open(RESULTS_FILE, "w", encoding="utf-8") as f:
        json.dump({
            "timestamp": TIMESTAMP,
            "judge_model": JUDGE_MODEL,
            "n_questions": len(test_cases),
            "n_configs": len(results),
            "results": results,
        }, f, indent=2, ensure_ascii=False)

    print(f"\nResults saved → {RESULTS_FILE}")

    # CSV export — one row per question per config
    csv_cols = ["q_num", "category", "llm", "chunk", "k", "weights",
                "actionability", "correctness", "faithfulness",
                "answer_relevancy", "contextual_recall", "latency"]
    with open(RESULTS_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=csv_cols)
        w.writeheader()
        for r in results:
            cfg = r["config"]
            llm     = _short(cfg["llm_model"], 20)
            chunk   = cfg["chunk_size"]
            k       = cfg["k"]
            weights = "pure-sem" if cfg["weights"] == [0.0, 1.0] else "70/30"
            for q_num, pq in enumerate(r["per_question"], 1):
                def _r(v): return round(v, 4) if isinstance(v, float) else v
                w.writerow({
                    "q_num":             q_num,
                    "category":          pq.get("category", ""),
                    "llm":               llm,
                    "chunk":             chunk,
                    "k":                 k,
                    "weights":           weights,
                    "actionability":     _r(pq["actionability"]),
                    "correctness":       _r(pq["correctness"]),
                    "faithfulness":      _r(pq["faithfulness"]),
                    "answer_relevancy":  _r(pq["answer_relevancy"]),
                    "contextual_recall": _r(pq["contextual_recall"]),
                    "latency":           _r(pq["latency"]),
                })
    print(f"CSV saved      → {RESULTS_CSV}")

    print_ranking_table(results)
    print_token_usage(judge_label)


    # Clean up checkpoint on successful completion
    if os.path.exists(CHECKPOINT_FILE):
        os.remove(CHECKPOINT_FILE)
        print("Checkpoint cleared.")


if __name__ == "__main__":
    main()
