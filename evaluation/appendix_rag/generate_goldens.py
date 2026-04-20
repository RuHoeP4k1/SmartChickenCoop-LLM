"""
generate_goldens.py — Phase B: Synthesise goldens from test_docs/

Uses DeepEval Synthesizer with gpt-4o-mini to generate question + ground-truth
pairs from the knowledge base. Chunk settings mirror production (1000, overlap 120).
Evolution mix targets the Q&A types most relevant to poultry husbandry eval.

Evolution weights (must sum to 1.0):
  REASONING      25%  — requires inference across facts
  CONCRETIZING   25%  — asks for specific thresholds / values
  COMPARATIVE    20%  — compares conditions or approaches
  CONSTRAINED    15%  — adds limiting conditions ("only if …")
  MULTICONTEXT   15%  — needs multiple document chunks

Output: evaluation/goldens_synth_v1.json
Format: list of {question, ground_truth, category, context, metadata} dicts
        directly loadable by evaluate_rag_ablation.py --goldens-file.

Usage:
    python evaluation/generate_goldens.py               # generate 50 goldens
    python evaluation/generate_goldens.py --n 10        # smoke test
    python evaluation/generate_goldens.py --spot-check  # also print 20 samples
"""

import os
import sys
import json
import random
import argparse
import glob as _glob

# Disable DeepEval telemetry
os.environ["DEEPEVAL_TELEMETRY_OPT_OUT"] = "YES"

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _ROOT)

from dotenv import load_dotenv
load_dotenv(dotenv_path=os.path.join(_ROOT, ".env"))

from deepeval.synthesizer import Synthesizer
from deepeval.synthesizer.config import (
    ContextConstructionConfig,
    EvolutionConfig,
    FiltrationConfig,
)
from deepeval.synthesizer.types import Evolution

TARGET_N        = 50
SYNTH_MODEL     = "gpt-4o-mini"
OUTPUT_PATH     = os.path.join(_HERE, "data", "goldens_synth_v1.json")

EVOLUTION_WEIGHTS = {
    Evolution.REASONING:    0.25,
    Evolution.CONCRETIZING: 0.25,
    Evolution.COMPARATIVE:  0.20,
    Evolution.CONSTRAINED:  0.15,
    Evolution.MULTICONTEXT: 0.15,
}

# ContextConstructionConfig.chunk_size is in tokens; 1000 matches the intent
# of production's 1000-char splits (deepeval's tokenizer is char-approximate).
CONTEXT_CFG = ContextConstructionConfig(
    chunk_size=1000,
    chunk_overlap=120,
    max_contexts_per_document=3,
    min_contexts_per_document=1,
    max_context_length=3,
    min_context_length=1,
    context_quality_threshold=0.5,
)

EVOLUTION_CFG = EvolutionConfig(
    num_evolutions=1,
    evolutions=EVOLUTION_WEIGHTS,
)

FILTRATION_CFG = FiltrationConfig(
    synthetic_input_quality_threshold=0.5,
    max_quality_retries=2,
)


def _doc_paths() -> list[str]:
    return sorted(_glob.glob(os.path.join(_ROOT, "test_docs", "*.pdf")))


def _golden_to_dict(g) -> dict:
    meta = g.additional_metadata or {}
    return {
        "question":     g.input,
        "ground_truth": g.expected_output or "",
        "category":     "synthetic",
        "context":      g.context or [],
        "metadata": {
            "source_file":    g.source_file,
            "evolutions":     meta.get("evolutions", []),
            "input_quality":  meta.get("synthetic_input_quality"),
            "context_quality": meta.get("context_quality"),
        },
    }


def generate(n_target: int = TARGET_N, spot_check: bool = False):
    doc_paths = _doc_paths()
    if not doc_paths:
        raise FileNotFoundError(f"No PDFs found in {os.path.join(_ROOT, 'test_docs')}")

    print(f"Synthesizer config:")
    print(f"  model:       {SYNTH_MODEL}")
    print(f"  docs:        {len(doc_paths)} PDFs")
    print(f"  chunk:       size=1000  overlap=120")
    print(f"  evolutions:  {', '.join(f'{e.name}={w:.0%}' for e, w in EVOLUTION_WEIGHTS.items())}")
    print(f"  target:      {n_target} goldens")
    print()

    # max_goldens_per_context chosen so 24 docs × 3 contexts × value ≈ well above target
    # to absorb quality filtering; we'll truncate to n_target afterward.
    max_goldens_per_context = max(2, -(-n_target // (len(doc_paths) * 2)))  # ceil division

    synthesizer = Synthesizer(
        model=SYNTH_MODEL,
        async_mode=True,
        max_concurrent=10,
        evolution_config=EVOLUTION_CFG,
        filtration_config=FILTRATION_CFG,
    )

    print(f"Generating goldens (max_goldens_per_context={max_goldens_per_context})…")
    goldens = synthesizer.generate_goldens_from_docs(
        document_paths=doc_paths,
        include_expected_output=True,
        max_goldens_per_context=max_goldens_per_context,
        context_construction_config=CONTEXT_CFG,
    )

    print(f"Raw goldens generated: {len(goldens)}")

    # Sort descending by input_quality so highest-quality ones are kept
    def _quality(g):
        meta = g.additional_metadata or {}
        q = meta.get("synthetic_input_quality")
        return q if q is not None else 0.0

    # Drop goldens from non-English source PDFs (e.g. BaroMIJTer is Dutch)
    EXCLUDED_SOURCES = {"baromijter"}
    goldens = [
        g for g in goldens
        if not any(ex in (g.source_file or "").lower() for ex in EXCLUDED_SOURCES)
    ]
    print(f"After language filter (excluded: {EXCLUDED_SOURCES}): {len(goldens)}")

    goldens_sorted = sorted(goldens, key=_quality, reverse=True)
    selected = goldens_sorted[:n_target]

    print(f"Selected after quality sort: {len(selected)}")

    rows = [_golden_to_dict(g) for g in selected]

    payload = {
        "version": "v1",
        "model": SYNTH_MODEL,
        "n": len(rows),
        "chunk_size": 1000,
        "chunk_overlap": 120,
        "evolution_weights": {e.name: w for e, w in EVOLUTION_WEIGHTS.items()},
        "goldens": rows,
    }

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(f"\nSaved {len(rows)} goldens -> {OUTPUT_PATH}")

    if spot_check:
        _print_spot_check(rows)

    return rows


def _print_spot_check(rows: list[dict], n: int = 20):
    sample = random.sample(rows, min(n, len(rows)))
    print("\n" + "=" * 80)
    print(f"SPOT-CHECK — {len(sample)} random goldens")
    print("=" * 80)
    for i, row in enumerate(sample, 1):
        print(f"\n[{i}/{len(sample)}]  evolution={row['metadata'].get('evolutions')}  "
              f"quality={row['metadata'].get('input_quality')}")
        print(f"  Q: {row['question']}")
        print(f"  A: {row['ground_truth'][:200]}{'…' if len(row['ground_truth']) > 200 else ''}")
        print(f"  src: {os.path.basename(row['metadata'].get('source_file') or '')}")


def spot_check_existing():
    with open(OUTPUT_PATH, encoding="utf-8") as f:
        payload = json.load(f)
    rows = payload["goldens"]
    print(f"Loaded {len(rows)} goldens from {OUTPUT_PATH}")
    _print_spot_check(rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic goldens from test_docs/")
    parser.add_argument("--n", type=int, default=TARGET_N,
                        help=f"Number of goldens to keep (default {TARGET_N})")
    parser.add_argument("--spot-check", action="store_true",
                        help="Print 20 random goldens after generation")
    parser.add_argument("--spot-check-only", action="store_true",
                        help="Skip generation, print 20 samples from existing file")
    args = parser.parse_args()

    if args.spot_check_only:
        spot_check_existing()
    else:
        generate(n_target=args.n, spot_check=args.spot_check)
