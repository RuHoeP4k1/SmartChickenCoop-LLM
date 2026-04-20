"""
compare_sensor_routing.py — Keyword vs LLM sensor routing comparison

Runs both methods on the same 19 sensor scenarios and produces a
side-by-side accuracy / latency / consistency comparison.

Usage:
    python evaluation/compare_sensor_routing.py --n-scenarios 4 --n-runs 1   # smoke
    python evaluation/compare_sensor_routing.py --n-runs 3 --verbose         # full
    python evaluation/compare_sensor_routing.py --model openrouter/ministral-14b-2512
"""

import os
import sys
import json
import time
import argparse
from datetime import datetime, timezone
from typing import Dict, List, Tuple, Optional

_HERE = os.path.dirname(os.path.abspath(__file__))
_EVAL = os.path.dirname(_HERE)
_ROOT = os.path.dirname(_EVAL)
sys.path.insert(0, _ROOT)
sys.path.insert(0, _EVAL)

from dotenv import load_dotenv
load_dotenv(dotenv_path=os.path.join(_ROOT, ".env"))

from backend.sensor_filter import should_include_sensors, format_sensor_summary
from evaluate_sensor_awareness import SENSOR_SCENARIOS, _fresh


# format_sensor_summary is now imported from backend.sensor_filter

# =============================================================================
# LLM CLASSIFIER
# =============================================================================

_CLASSIFIER_PROMPT = """\
You decide whether live coop sensor readings should be included when answering \
a chicken-keeping question. Return ONLY the word INCLUDE or EXCLUDE. No explanation.

RULES (apply in order, stop at first match):
1. User asks about their own coop conditions right now (uses words like \
"my coop", "right now", "how hot is it", "check on my chickens", "is my \
feeder okay", "did I close the door", reports symptoms like panting/coughing/\
lethargy, or asks for a status update) -> INCLUDE
2. User describes a behavioral symptom or health concern about their birds \
(molting, limping, diarrhea, sneezing, ruffled feathers, mites, etc.) AND \
the question is NOT purely encyclopedic (not "what causes...", "how to...", \
"why do chickens...") -> INCLUDE
3. User mentions air quality or smells (ammonia, stuffy, stinks) AND the \
question is NOT purely encyclopedic -> INCLUDE
4. Resource (feeder or waterer) is low or empty AND user mentions food, water, \
feeder, waterer, hungry, thirsty, or refilling -> INCLUDE
5. Any sensor reading is CRITICAL (temperature, humidity, heat stress, H2S, \
mold risk) AND the question is about chickens, coops, health, eggs, feed, \
water, disease, or illness -> INCLUDE
6. User asks about flock count, egg count, or door/ventilation status -> INCLUDE
7. For general knowledge questions (breed advice, "what is...", "how to...", \
lifecycle questions, "can chickens...") with no personal coop reference \
-> EXCLUDE

SENSOR STATUS: {sensor_summary}
USER QUESTION: {query}
DECISION:"""

_classifier_llm = None
_classifier_model_id = None


def _get_classifier_llm(model: str = None):
    """Create a dedicated LLM instance for classification (low temp, tiny output)."""
    global _classifier_llm, _classifier_model_id
    if model is None:
        model = os.getenv("OLLAMA_MODEL", "smollm2:1.7b")
    if _classifier_llm is None or _classifier_model_id != model:
        if model.startswith("openrouter/"):
            from langchain_openai import ChatOpenAI
            _classifier_llm = ChatOpenAI(
                model=model.removeprefix("openrouter/"),
                base_url="https://openrouter.ai/api/v1",
                api_key=os.getenv("OPENROUTER_API_KEY"),
                temperature=0.1,
                max_tokens=10,
            )
        else:
            from langchain_ollama import OllamaLLM
            _classifier_llm = OllamaLLM(
                model=model,
                temperature=0.1,
                num_predict=10,
            )
        _classifier_model_id = model
    return _classifier_llm


def llm_classify_sensors(
    query: str,
    sensor_data: Dict,
    model: str = None,
) -> Tuple[Optional[bool], float, str]:
    """
    Ask the LLM whether sensors should be included.

    Returns: (include_decision, latency_ms, raw_response)
        include_decision is None on parse error.
    """
    llm = _get_classifier_llm(model)
    summary = format_sensor_summary(sensor_data)
    prompt = _CLASSIFIER_PROMPT.format(sensor_summary=summary, query=query)

    t0 = time.perf_counter()
    response = llm.invoke(prompt)
    latency_ms = (time.perf_counter() - t0) * 1000

    # Handle both ChatOpenAI (AIMessage) and OllamaLLM (string)
    raw = response.content if hasattr(response, "content") else str(response)
    raw = raw.strip()

    # Extract token usage if available (OpenRouter / ChatOpenAI)
    token_usage = None
    if hasattr(response, "response_metadata"):
        token_usage = response.response_metadata.get("token_usage")

    # Permissive parse
    upper = raw.upper()
    if upper.startswith("INCL"):
        decision = True
    elif upper.startswith("EXCL"):
        decision = False
    else:
        decision = None  # parse error

    return decision, latency_ms, raw, token_usage


# =============================================================================
# COMPARISON RUNNER
# =============================================================================

def run_comparison(
    n_runs: int = 3,
    model: str = None,
    n_scenarios: int = None,
    verbose: bool = False,
) -> Dict:
    """Run keyword and LLM routing on all scenarios, return results dict."""

    scenarios = SENSOR_SCENARIOS[:n_scenarios] if n_scenarios else SENSOR_SCENARIOS
    is_openrouter = (model or os.getenv("OLLAMA_MODEL", "")).startswith("openrouter/")

    display_model = model or os.getenv("OLLAMA_MODEL") or "(OLLAMA_MODEL not set)"
    print("=" * 80)
    print("SENSOR ROUTING COMPARISON: Keyword vs LLM Classifier")
    print(f"  Model: {display_model}")
    print(f"  Scenarios: {len(scenarios)} | Runs per scenario: {n_runs}")
    print("=" * 80)

    results = []

    for i, sc in enumerate(scenarios):
        sid = sc["id"]
        expected = sc["expect_sensor_in"]
        sensor_data = _fresh(sc["sensor_data"])
        query = sc["question"]

        # --- Keyword method (deterministic, run once) ---
        t0 = time.perf_counter()
        kw_decision = should_include_sensors(query, sensor_data)
        kw_latency_ms = (time.perf_counter() - t0) * 1000

        # --- LLM method (run n_runs times) ---
        llm_runs = []
        for run_i in range(n_runs):
            decision, lat_ms, raw, tok = llm_classify_sensors(query, sensor_data, model)
            llm_runs.append({
                "decision": decision,
                "latency_ms": round(lat_ms, 1),
                "raw_response": raw,
                "token_usage": tok,
            })
            if is_openrouter and run_i < n_runs - 1:
                time.sleep(0.5)

        # Majority vote (None = parse error counts as EXCLUDE)
        votes = [r["decision"] if r["decision"] is not None else False for r in llm_runs]
        include_count = sum(1 for v in votes if v)
        llm_majority = include_count > n_runs / 2

        # Consistency: all runs agree (ignoring parse errors)
        valid = [r["decision"] for r in llm_runs if r["decision"] is not None]
        llm_consistent = len(set(valid)) <= 1 if valid else False

        parse_errors = sum(1 for r in llm_runs if r["decision"] is None)
        mean_lat = sum(r["latency_ms"] for r in llm_runs) / n_runs

        entry = {
            "id": sid,
            "label": sc["label"],
            "question": query,
            "expected": expected,
            "keyword": {
                "decision": kw_decision,
                "correct": kw_decision == expected,
                "latency_ms": round(kw_latency_ms, 3),
            },
            "llm": {
                "runs": llm_runs,
                "majority": llm_majority,
                "correct": llm_majority == expected,
                "consistent": llm_consistent,
                "parse_errors": parse_errors,
                "mean_latency_ms": round(mean_lat, 1),
            },
        }
        results.append(entry)

        if verbose:
            kw_ok = "OK" if entry["keyword"]["correct"] else "FAIL"
            llm_ok = "OK" if entry["llm"]["correct"] else "FAIL"
            llm_decisions = " ".join(
                "INCL" if r["decision"] else ("EXCL" if r["decision"] is not None else "ERR")
                for r in llm_runs
            )
            print(f"  {sid} | Kw={kw_ok} LLM={llm_ok} [{llm_decisions}] | {mean_lat:.0f}ms | {query[:50]}")

        # Rate limit between scenarios for OpenRouter
        if is_openrouter and i < len(scenarios) - 1:
            time.sleep(0.3)

    return {
        "model": display_model,
        "n_runs": n_runs,
        "n_scenarios": len(scenarios),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "scenarios": results,
    }


# =============================================================================
# OUTPUT
# =============================================================================

def print_results_table(data: Dict):
    """Print per-scenario table and summary."""
    results = data["scenarios"]
    n_runs = data["n_runs"]

    print()
    print("=" * 80)
    print("RESULTS")
    print("=" * 80)

    # Header
    run_cols = "".join(f" | LLM {i+1} " for i in range(n_runs))
    header = f"{'ID':<4} | {'Question':<42} | {'Expect':<7} | {'Keyw':<7}{run_cols} | {'Kw':<4} | {'LLM':<4} | {'Lat ms':<7}"
    print(header)
    print("-" * len(header))

    kw_correct = 0
    llm_correct = 0
    llm_consistent = 0
    total_llm_lat = 0

    for r in results:
        exp = "INCL" if r["expected"] else "EXCL"
        kw = "INCL" if r["keyword"]["decision"] else "EXCL"
        kw_mark = " OK " if r["keyword"]["correct"] else "FAIL"
        llm_mark = " OK " if r["llm"]["correct"] else "FAIL"

        run_strs = ""
        for run in r["llm"]["runs"]:
            d = run["decision"]
            if d is None:
                run_strs += " |  ERR  "
            elif d:
                run_strs += " |  INCL "
            else:
                run_strs += " |  EXCL "

        q = r["question"][:42]
        lat = f"{r['llm']['mean_latency_ms']:.0f}"

        print(f"{r['id']:<4} | {q:<42} | {exp:<7} | {kw:<7}{run_strs} | {kw_mark} | {llm_mark} | {lat:>7}")

        if r["keyword"]["correct"]:
            kw_correct += 1
        if r["llm"]["correct"]:
            llm_correct += 1
        if r["llm"]["consistent"]:
            llm_consistent += 1
        total_llm_lat += r["llm"]["mean_latency_ms"]

    n = len(results)
    print()
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"  {'Metric':<30} {'Keyword':<15} {'LLM (majority)':<15}")
    print(f"  {'-'*30} {'-'*15} {'-'*15}")
    print(f"  {'Accuracy':<30} {kw_correct}/{n:<15} {llm_correct}/{n:<15}")
    print(f"  {'Mean latency (ms)':<30} {'<1':<15} {total_llm_lat/n:.0f}")
    print(f"  {'Consistency (all runs agree)':<30} {n}/{n:<15} {llm_consistent}/{n:<15}")

    # Disagreements
    disagreements = [r for r in results if r["keyword"]["correct"] != r["llm"]["correct"]]
    if disagreements:
        print()
        print("DISAGREEMENTS (one correct, other wrong):")
        for r in disagreements:
            kw_ok = "OK" if r["keyword"]["correct"] else "FAIL"
            llm_ok = "OK" if r["llm"]["correct"] else "FAIL"
            print(f"  {r['id']} — Keyword={kw_ok}, LLM={llm_ok} — {r['question']}")

    # Both wrong
    both_wrong = [r for r in results if not r["keyword"]["correct"] and not r["llm"]["correct"]]
    if both_wrong:
        print()
        print("BOTH WRONG:")
        for r in both_wrong:
            print(f"  {r['id']} — {r['question']}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Compare keyword vs LLM sensor routing")
    parser.add_argument("--model", type=str, default=None,
                        help="LLM model (default: OLLAMA_MODEL env var)")
    parser.add_argument("--n-runs", type=int, default=3,
                        help="Number of LLM runs per scenario (default: 3)")
    parser.add_argument("--n-scenarios", type=int, default=None,
                        help="Limit to first N scenarios (default: all 19)")
    parser.add_argument("--verbose", action="store_true",
                        help="Print per-scenario results during run")
    args = parser.parse_args()

    data = run_comparison(
        n_runs=args.n_runs,
        model=args.model,
        n_scenarios=args.n_scenarios,
        verbose=args.verbose,
    )

    print_results_table(data)

    # Save JSON
    results_dir = os.path.join(_EVAL, "results")
    os.makedirs(results_dir, exist_ok=True)
    out_path = os.path.join(results_dir, "sensor_routing_comparison.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
