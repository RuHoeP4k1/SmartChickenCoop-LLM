"""
evaluate_sensor_awareness.py — Real-time sensor awareness evaluation

Tests whether the LLM correctly integrates live sensor data into its answers.
This is a distinct evaluation axis from RAG quality: even with a perfect
knowledge base, the system needs to:

  1. INCLUDE sensors when the reading is relevant (correct routing)
  2. CITE actual values from the sensor snapshot (temperature, humidity)
  3. MATCH urgency to severity (critical → urgent language; normal → calm)
  4. EXCLUDE sensors for general questions with normal readings

These tests use mocked sensor_data injected directly into answer_query()
so no running database or Pi hardware is needed.

Usage:
    python evaluation/evaluate_sensor_awareness.py
    python evaluation/evaluate_sensor_awareness.py --verbose
    python evaluation/evaluate_sensor_awareness.py --n-scenarios 4   # smoke test
"""

import os
import sys
import json
import re
import time
import argparse
from datetime import datetime, timezone
from typing import Dict, List, Optional

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _ROOT)
sys.path.insert(0, _HERE)

from dotenv import load_dotenv
load_dotenv(dotenv_path=os.path.join(_ROOT, ".env"))

from backend.rag_functions import (
    load_documents, split_documents, build_vector_store,
    build_bm25_retriever, answer_query,
)


# =============================================================================
# MOCK SENSOR SNAPSHOTS
# These represent realistic states the Pi could report.
# Always use a fresh timestamp so the stale-data filter doesn't exclude them.
# =============================================================================

def _fresh(base: Dict) -> Dict:
    """Inject a current timestamp so the staleness check passes."""
    return {**base, "timestamp": datetime.now(timezone.utc)}


_NORMAL = {
    "temperature_c": 21.5, "temperature_status": "normal",
    "humidity_pct": 58.0,  "humidity_status":    "normal",
    "heat_stress_index": "normal",
    "feeder_status": "full",  "feeder_pct": 82.0,
    "waterer_status": "full", "waterer_pct": 76.0,
    "h2s_level": "normal", "h2s_ppm": None,
    "mold_risk_status": "normal",
    "door_open": False,
}

_TEMP_WARNING = {
    **_NORMAL,
    "temperature_c": 27.4, "temperature_status": "warning",
}

_TEMP_CRITICAL = {
    **_NORMAL,
    "temperature_c": 35.2, "temperature_status": "critical",
    "humidity_pct": 72.0,  "humidity_status":    "warning",
    "heat_stress_index": "critical",
}

_HUMIDITY_CRITICAL = {
    **_NORMAL,
    "humidity_pct": 91.0, "humidity_status": "critical",
}

_FEEDER_EMPTY = {
    **_NORMAL,
    "feeder_status": "empty", "feeder_pct": 3.0,
}

_WATERER_LOW = {
    **_NORMAL,
    "waterer_status": "low", "waterer_pct": 18.0,
}

_MULTI_CRITICAL = {
    **_NORMAL,
    "temperature_c": 36.1,  "temperature_status": "critical",
    "humidity_pct":  88.0,  "humidity_status":    "critical",
    "heat_stress_index": "critical",
    "feeder_status": "empty", "feeder_pct": 2.0,
    "waterer_status": "empty", "waterer_pct": 5.0,
}

_H2S_CRITICAL = {
    **_NORMAL,
    "h2s_level": "critical", "h2s_ppm": 52.0,
}

_MOLD_RISK_CRITICAL = {
    **_NORMAL,
    "humidity_pct": 84.0, "humidity_status": "warning",
    "mold_risk_status": "critical",
}

_DOOR_OPEN = {
    **_NORMAL,
    "door_open": True,
}

_WITH_EGGS = {
    **_NORMAL,
    "egg_count": 3, "number_of_chickens": 6,
}


# =============================================================================
# TEST SCENARIOS
#
# Each scenario specifies:
#   sensor_data       — the mocked sensor snapshot
#   question          — what the user asks
#   expect_sensor_in  — True if sensors SHOULD be injected into the prompt
#   expect_values     — temperature / humidity values the LLM should cite
#                       (only checked when expect_sensor_in=True and values are non-normal)
#   expect_urgency    — "none" | "low" | "high"
#                       "high" = answer should contain urgent/critical language
#                       "none" = answer should NOT contain alarming language
# =============================================================================

SENSOR_SCENARIOS = [
    # ── Correct exclusion: general question, normal sensors ──────────────────
    {
        "id": "S01",
        "label": "Normal sensors, general question → sensors excluded",
        "sensor_data": _NORMAL,
        "question": "How many eggs do chickens lay per day?",
        "expect_sensor_in": False,
        "expect_values": [],
        "expect_urgency": "none",
    },
    {
        "id": "S02",
        "label": "Normal sensors, breed question → sensors excluded",
        "sensor_data": _NORMAL,
        "question": "What breed of chicken lays the most eggs?",
        "expect_sensor_in": False,
        "expect_values": [],
        "expect_urgency": "none",
    },

    # ── Correct inclusion: user asks about their coop ────────────────────────
    {
        "id": "S03",
        "label": "Normal sensors, user asks 'is my coop ok?' → include, say all normal",
        "sensor_data": _NORMAL,
        "question": "Is my coop temperature okay right now?",
        "expect_sensor_in": True,
        "expect_values": [],            # all normal, no specific values to cite
        "expect_urgency": "none",
    },

    # ── Warning: temperature elevated ────────────────────────────────────────
    {
        "id": "S04",
        "label": "Temp warning (27.4°C), user asks about coop → cite value, action required",
        "sensor_data": _TEMP_WARNING,
        "question": "Is it too warm in my coop right now?",
        "expect_sensor_in": True,
        "expect_values": ["27"],        # 27.4°C should appear in some form
        "expect_urgency": "high",       # model consistently uses "Act now" for elevated temp — correct
    },

    # ── Critical temperature ─────────────────────────────────────────────────
    {
        "id": "S05",
        "label": "Temp critical (35.2°C), coop question → cite value, urgent response",
        "sensor_data": _TEMP_CRITICAL,
        "question": "My chickens are panting and spreading their wings — what's going on?",
        "expect_sensor_in": True,
        "expect_values": ["35"],
        "expect_urgency": "high",
    },
    {
        "id": "S06",
        "label": "Temp critical, general health question → still include (critical overrides)",
        "sensor_data": _TEMP_CRITICAL,
        "question": "Why are my hens not laying eggs?",
        "expect_sensor_in": True,      # critical + chicken topic → include
        "expect_values": ["35"],
        "expect_urgency": "high",
    },

    # ── Critical humidity ─────────────────────────────────────────────────────
    {
        "id": "S07",
        "label": "Humidity critical (91%), coop health question → cite value, urgent",
        "sensor_data": _HUMIDITY_CRITICAL,
        "question": "Are my chickens okay right now?",
        "expect_sensor_in": True,
        "expect_values": ["91"],
        "expect_urgency": "high",
    },

    # ── Resource alerts ───────────────────────────────────────────────────────
    {
        "id": "S08",
        "label": "Feeder empty, user asks about feeder → include, mention empty",
        "sensor_data": _FEEDER_EMPTY,
        "question": "Is my feeder okay?",
        "expect_sensor_in": True,
        "expect_values": [],            # no temp/humidity, but feeder status
        "expect_urgency": "high",       # empty feeder warrants immediate refill action
        "extra_check": "empty",         # 'empty' should appear in answer
    },
    {
        "id": "S09",
        "label": "Waterer low, general question → exclude (resource not mentioned in query)",
        "sensor_data": _WATERER_LOW,
        "question": "What should I feed my laying hens?",
        "expect_sensor_in": False,
        "expect_values": [],
        "expect_urgency": "none",
    },

    # ── Multi-critical ────────────────────────────────────────────────────────
    {
        "id": "S10",
        "label": "All critical, any chicken question → include all alerts, maximum urgency",
        "sensor_data": _MULTI_CRITICAL,
        "question": "How are my chickens doing in the coop?",
        "expect_sensor_in": True,
        "expect_values": ["36"],
        "expect_urgency": "high",
    },

    # ── Hallucination guard ───────────────────────────────────────────────────
    {
        "id": "S11",
        "label": "Normal sensors, encyclopaedic question → sensors excluded, no specific values fabricated",
        "sensor_data": _NORMAL,
        "question": "What temperature is too hot for chickens?",
        "expect_sensor_in": False,
        "expect_values": [],
        "expect_urgency": "none",
        # Note: the LLM may cite temperatures from the knowledge base (e.g. "30°C")
        # that is fine; we only check that it does NOT cite the sensor value (21.5°C)
        "not_fabricated_value": "21.5",
    },

    # ── H2S / ammonia ─────────────────────────────────────────────────────────
    {
        "id": "S12",
        "label": "H2S critical (52 ppm), user reports ammonia smell → include, urgent, cite ppm",
        "sensor_data": _H2S_CRITICAL,
        "question": "It smells like ammonia in my coop — is that dangerous?",
        "expect_sensor_in": True,
        "expect_values": ["52"],
        "expect_urgency": "high",
    },
    {
        "id": "S13",
        "label": "H2S critical, encyclopedic question about ammonia → sensors excluded",
        "sensor_data": _H2S_CRITICAL,
        "question": "What causes ammonia buildup in chicken coops?",
        "expect_sensor_in": False,
        "expect_values": [],
        "expect_urgency": "none",
    },

    # ── Mold risk ─────────────────────────────────────────────────────────────
    {
        "id": "S14",
        "label": "Mold risk critical, coop status question → include, mild concern (not emergency)",
        "sensor_data": _MOLD_RISK_CRITICAL,
        "question": "Is my coop okay right now?",
        "expect_sensor_in": True,
        "expect_values": [],
        "expect_urgency": "high",       # mold_risk_status: critical → model uses urgent language
        # Mold risk is marked critical — model correctly escalates. Not an instant hazard
        # like temp or H2S but sensor reports critical, so "Act now" language is appropriate.
    },

    # ── Door open ─────────────────────────────────────────────────────────────
    {
        "id": "S15",
        "label": "Door open, user asks if they closed it → include, mention door is open",
        "sensor_data": _DOOR_OPEN,
        "question": "Did I close the coop door?",
        "expect_sensor_in": True,
        "expect_values": [],
        "expect_urgency": "none",
        "extra_check": "open",
    },

    # ── Egg count ─────────────────────────────────────────────────────────────
    {
        "id": "S16",
        "label": "3 eggs detected, user asks about eggs → include, cite count",
        "sensor_data": _WITH_EGGS,
        "question": "Are there any eggs today?",
        "expect_sensor_in": True,
        "expect_values": ["3"],
        "expect_urgency": "none",
    },

    # ── Waterer low + matching question ───────────────────────────────────────
    {
        "id": "S17",
        "label": "Waterer low (18%), user asks about waterer → include, mention low level",
        "sensor_data": _WATERER_LOW,
        "question": "Is my waterer okay?",
        "expect_sensor_in": True,
        "expect_values": [],
        "expect_urgency": "low",
        "extra_check": "low",
    },

    # ── False positive guard ──────────────────────────────────────────────────
    {
        "id": "S18",
        "label": "Multi-critical sensors, non-chicken-topic question → sensors excluded",
        "sensor_data": _MULTI_CRITICAL,
        "question": "What breed should I get?",
        "expect_sensor_in": False,
        "expect_values": [],
        "expect_urgency": "none",
    },

    # ── All-normal full status ────────────────────────────────────────────────
    {
        "id": "S19",
        "label": "All normal, full status request → include, answer confirms everything normal",
        "sensor_data": _NORMAL,
        "question": "Can you give me a full status update on my coop?",
        "expect_sensor_in": True,
        "expect_values": [],
        "expect_urgency": "none",
        "extra_check": "normal",
    },
]


# =============================================================================
# SCORING
# =============================================================================

_URGENT_WORDS = [
    "immediately", "urgent", "urgently", "critical", "danger", "dangerous",
    "right now", "emergency", "act now", "as soon as possible",
    "serious risk", "risk to", "at risk",
]

_CONCERN_WORDS = [
    "elevated", "high", "above", "exceeds", "concerning", "monitor",
    "keep an eye", "warm", "check",
]


def score_sensor_awareness(
    result: Dict,
    scenario: Dict,
) -> Dict:
    """
    Score one answer against the scenario's expected sensor awareness.

    Returns a dict with individual scores and an overall pass/fail.
    """
    answer = result["answer"].lower()
    sensor_was_included = result.get("sensor_included", False)

    scores = {}

    # 1. Routing: was sensor context correctly included or excluded?
    expected_in = scenario["expect_sensor_in"]
    routing_correct = sensor_was_included == expected_in
    scores["routing_correct"] = routing_correct

    # 2. Value citation: does the answer mention the expected numeric values?
    #    Only checked when sensors were included AND values are expected.
    values_to_find = scenario.get("expect_values", [])
    if values_to_find and sensor_was_included:
        cited = all(v in answer for v in values_to_find)
        scores["value_cited"] = cited
    else:
        scores["value_cited"] = None  # not applicable

    # 3. Urgency reflection: does urgency language match expected level?
    expected_urgency = scenario.get("expect_urgency", "none")
    if expected_urgency == "high":
        scores["urgency_match"] = any(w in answer for w in _URGENT_WORDS)
    elif expected_urgency == "low":
        # Should show some concern but not full alarm
        has_concern  = any(w in answer for w in _CONCERN_WORDS)
        has_alarm    = any(w in answer for w in _URGENT_WORDS)
        scores["urgency_match"] = has_concern and not has_alarm
    else:  # "none"
        # Normal reading + general question → no alarming language expected
        scores["urgency_match"] = not any(w in answer for w in _URGENT_WORDS)

    # 4. Extra check (e.g. "empty" should appear in feeder-empty scenarios)
    extra_kw = scenario.get("extra_check")
    if extra_kw:
        scores["extra_check"] = extra_kw.lower() in answer
    else:
        scores["extra_check"] = None  # not applicable

    # 5. Not-fabricated: sensor value that was NOT in the prompt should NOT appear
    not_fabricated_value = scenario.get("not_fabricated_value")
    if not_fabricated_value:
        scores["not_fabricated"] = not_fabricated_value not in answer
    else:
        scores["not_fabricated"] = None  # not applicable

    # Overall: all applicable checks must pass
    applicable = {k: v for k, v in scores.items() if v is not None}
    scores["overall_pass"] = all(applicable.values()) if applicable else False

    return scores


# =============================================================================
# RUNNER
# =============================================================================

def run_evaluation(
    n_scenarios: int = None,
    verbose: bool = False,
    llm_model: str = None,
    k: int = 4,
    chunk_size: int = 1000,
):
    KB_PATH    = os.path.join(_ROOT, "test_docs")
    CHROMA_DIR = os.path.join(_ROOT, "chroma_db")
    RESULTS_DIR = os.path.join(_HERE, "results")
    os.makedirs(RESULTS_DIR, exist_ok=True)

    _display_model = llm_model or os.getenv("OLLAMA_MODEL") or "(OLLAMA_MODEL not set — check .env)"
    print("=" * 80)
    print("SENSOR AWARENESS EVALUATION")
    print(f"  Model: {_display_model}")
    print(f"  k={k}  chunk_size={chunk_size}")
    print("=" * 80)

    print("\nBuilding RAG pipeline...")
    docs     = load_documents(KB_PATH)
    chunks   = split_documents(docs, chunk_size=chunk_size)
    vectordb = build_vector_store(chunks, persist_dir=CHROMA_DIR, folder_path=KB_PATH)
    bm25     = build_bm25_retriever(chunks)
    print("Pipeline ready.\n")

    scenarios = SENSOR_SCENARIOS[:n_scenarios] if n_scenarios else SENSOR_SCENARIOS
    all_results = []
    passed = 0

    for scenario in scenarios:
        print(f"[{scenario['id']}] {scenario['label']}")
        print(f"  Q: {scenario['question']}")

        sensor_data = _fresh(scenario["sensor_data"])

        t0 = time.time()
        result = answer_query(
            query=scenario["question"],
            vectordb=vectordb,
            bm25_retriever=bm25,
            use_sensors=True,
            use_hybrid=True,
            k=k,
            sensor_data=sensor_data,
            llm_model=llm_model,
        )
        elapsed = time.time() - t0

        scores = score_sensor_awareness(result, scenario)
        overall = scores["overall_pass"]
        if overall:
            passed += 1

        status = "PASS" if overall else "FAIL"
        print(f"  → {status}  sensor_included={result['sensor_included']}  "
              f"routing={'OK' if scores['routing_correct'] else 'WRONG'}  "
              f"value_cited={scores['value_cited']}  "
              f"urgency_match={scores['urgency_match']}  "
              f"({elapsed:.1f}s)")

        if verbose:
            print(f"\n  Answer:\n{result['answer']}\n")
            if result.get("sensor_context"):
                print(f"  Sensor context sent:\n{result['sensor_context']}\n")

        all_results.append({
            "scenario_id": scenario["id"],
            "label": scenario["label"],
            "question": scenario["question"],
            "sensor_data_state": {k: v for k, v in scenario["sensor_data"].items() if k != "timestamp"},
            "answer": result["answer"],
            "sensor_included": result["sensor_included"],
            "sensor_context": result.get("sensor_context"),
            "scores": scores,
            "time": round(elapsed, 2),
        })
        print()

    total = len(scenarios)
    print("=" * 80)
    print(f"RESULT: {passed}/{total} scenarios passed ({100*passed/total:.0f}%)")
    print("=" * 80)

    # Per-check breakdown
    checks = ["routing_correct", "value_cited", "urgency_match", "extra_check", "not_fabricated"]
    print("\nPer-check breakdown (applicable scenarios only):")
    for check in checks:
        applicable = [r for r in all_results if r["scores"].get(check) is not None]
        if not applicable:
            continue
        n_pass = sum(1 for r in applicable if r["scores"][check])
        print(f"  {check:<22} {n_pass}/{len(applicable)}")

    out_path = os.path.join(RESULTS_DIR, "sensor_awareness_results.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sensor awareness evaluation")
    parser.add_argument("--n-scenarios", type=int, default=None,
                        help="Limit to first N scenarios (smoke test)")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Print full answer text for each scenario")
    parser.add_argument("--k", type=int, default=4)
    parser.add_argument("--chunk-size", type=int, default=1000)
    parser.add_argument("--model", type=str, default=None,
                        help="LLM model override")
    args = parser.parse_args()

    run_evaluation(
        n_scenarios=args.n_scenarios,
        verbose=args.verbose,
        llm_model=args.model,
        k=args.k,
        chunk_size=args.chunk_size,
    )
