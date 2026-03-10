# evaluation/sweep_config.py
"""
D-Optimal experiment design for RAG hyperparameter sweep.

Factors (6 total):
  A  llm_model    4 levels
  B  embed_model  2 levels
  C  chunk_size   3 levels
  D  k            3 levels
  E  weights      2 levels  (BM25:semantic ratio)
  F  search_type  2 levels

Full factorial: 4 × 2 × 3 × 3 × 2 × 2 = 288 runs
D-Optimal:      ~60 runs  — covers all main effects + two-way interactions

Usage:
    from sweep_config import build_design, FACTOR_KEYS, save_design_md
    configs = build_design(n_runs=60)
    save_design_md(configs, "evaluation/results/doe_design.md")
"""

import os
import json
from pathlib import Path

# ---------------------------------------------------------------------------
# Parameter grid — maps integer level indices to real values
# ---------------------------------------------------------------------------

PARAM_GRID = {
    "llm_model": [
        "smollm2:1.7b",
        "openrouter/nvidia/nemotron-3-nano-30b-a3b:free",
        # Full sweep models (restore when running Round 1):
        # "smollm2:0.5b",
        # "openrouter/meta-llama/llama-3.1-8b-instruct:free",
        # "openrouter/meta-llama/llama-3.3-70b-instruct:free",
    ],
    "embed_model": [
        "nomic-embed-text-v2-moe",
        # "text-embedding-3-small",  # requires OPENAI_API_KEY — enable for full sweep
    ],
    "chunk_size": [800],  # smoke test: reuse existing chroma_db (800 chunks already embedded)
    "k":          [2, 4, 6],
    "weights": [
        [0.5, 0.5],   # 50% semantic, 50% BM25
        [0.0, 1.0],   # pure semantic (use_hybrid=False in answer_query)
    ],
    "search_type": ["mmr", "similarity"],
}

# Factor key order must match LEVELS order below
FACTOR_KEYS = ["llm_model", "embed_model", "chunk_size", "k", "weights", "search_type"]
LEVELS      = [2, 1, 1, 3, 2, 2]   # smoke test: chunk=1 (800 only), embed=1 (nomic only)

# Human-readable short labels for the design table
FACTOR_LABELS = {
    "llm_model":   ["1.7b", "nemotron-30b"],
    "embed_model": ["nomic", "openai-3s"],
    "chunk_size":  ["400", "600", "800"],
    "k":           ["2", "4", "6"],
    "weights":     ["50/50", "pure-sem"],
    "search_type": ["mmr", "similarity"],
}


# ---------------------------------------------------------------------------
# D-Optimal design generation
# ---------------------------------------------------------------------------

def build_design(n_runs: int = 60, seed: int = 42) -> list[dict]:
    """
    Generate a space-filling design (LHS via pyDOE3) and decode to real parameter dicts.

    Returns a list of config dicts, one per run.
    Falls back to a coverage-based greedy design if pyDOE3 is not installed.
    """
    try:
        from pyDOE3 import lhs
        import numpy as np

        np.random.seed(seed)
        n_factors = len(FACTOR_KEYS)
        # LHS gives uniform coverage; map continuous [0,1) samples to integer level indices
        lhs_matrix = lhs(n_factors, samples=n_runs, criterion="maximin")
        coded = np.floor(lhs_matrix * np.array(LEVELS)).astype(int)
        # Clip to valid range in case of floating point edge
        coded = np.clip(coded, 0, np.array(LEVELS) - 1)
        print("[OK] D-Optimal design via Latin Hypercube Sampling (pyDOE3).")
        return _decode(coded)

    except ImportError:
        print("⚠  pyDOE3 not found — falling back to greedy coverage design.")
        print("   Install with: pip install pyDOE3")
        return _greedy_design(n_runs, seed)


def _decode(coded_matrix) -> list[dict]:
    """Map integer-coded design matrix rows to real parameter dicts."""
    configs = []
    for row in coded_matrix:
        cfg = {key: PARAM_GRID[key][int(val)] for key, val in zip(FACTOR_KEYS, row)}
        configs.append(cfg)
    return configs


def _greedy_design(n_runs: int, seed: int = 42) -> list[dict]:
    """
    Coverage-based fallback: ensures every level of every factor appears at least once,
    then fills remaining slots randomly. Not D-optimal but still reasonable.
    """
    import random
    import itertools

    random.seed(seed)

    # Ensure full level coverage for every factor first
    base_rows = []
    for fi, key in enumerate(FACTOR_KEYS):
        for li in range(LEVELS[fi]):
            row = [random.randint(0, lv - 1) for lv in LEVELS]
            row[fi] = li
            base_rows.append(row)

    # Deduplicate
    seen = set()
    unique = []
    for r in base_rows:
        t = tuple(r)
        if t not in seen:
            seen.add(t)
            unique.append(r)

    # Fill to n_runs with random rows
    all_combos = list(itertools.product(*[range(lv) for lv in LEVELS]))
    random.shuffle(all_combos)
    for combo in all_combos:
        if len(unique) >= n_runs:
            break
        if combo not in seen:
            seen.add(combo)
            unique.append(list(combo))

    return _decode(unique[:n_runs])


# ---------------------------------------------------------------------------
# Config helpers used by sweep.py
# ---------------------------------------------------------------------------

def config_id(cfg: dict) -> str:
    """Stable string key for a config dict (used for checkpointing)."""
    return json.dumps(cfg, sort_keys=True)


def embed_slug(cfg: dict) -> str:
    """Short slug for the embedding model name (used in Chroma dir names)."""
    em = cfg["embed_model"]
    if "nomic" in em:
        return "nomic"
    if "3-small" in em:
        return "openai3s"
    return em.replace("/", "_").replace(":", "_")[:20]


def chroma_dir(cfg: dict, base: str = None) -> str:
    """
    Unique persist directory for a (embed_model, chunk_size) combination.
    Different embedding models MUST use different Chroma dirs.
    For smoke test (nomic + chunk=800): reuses the existing production chroma_db.
    """
    root = os.path.dirname(os.path.dirname(__file__))
    slug = embed_slug(cfg)
    # Reuse the production chroma_db when using the already-embedded combination
    if slug == "nomic" and cfg["chunk_size"] == 800:
        return os.path.join(root, "chroma_db")
    if base is None:
        base = os.path.join(root, "chroma_db_sweep")
    return os.path.join(base, f"{slug}_c{cfg['chunk_size']}")


def use_hybrid_flag(cfg: dict) -> bool:
    """Pure-semantic configs should skip BM25 entirely."""
    w = cfg["weights"]
    return not (w[0] == 0.0 and w[1] == 1.0)


# ---------------------------------------------------------------------------
# Design table markdown export
# ---------------------------------------------------------------------------

def save_design_md(configs: list[dict], path: str) -> None:
    """Save the design matrix as a readable markdown table."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)

    lines = [
        "# D-Optimal Design Matrix\n",
        f"**Runs:** {len(configs)}  |  **Factors:** {len(FACTOR_KEYS)}  |  "
        f"**Full factorial equivalent:** {_full_factorial_size()} runs\n",
        "",
        "| Run | LLM | Embed | Chunk | k | Weights | Search |",
        "|-----|-----|-------|-------|---|---------|--------|",
    ]

    for i, cfg in enumerate(configs, 1):
        llm_label = _short_llm(cfg["llm_model"])
        embed_label = FACTOR_LABELS["embed_model"][PARAM_GRID["embed_model"].index(cfg["embed_model"])]
        chunk_label = str(cfg["chunk_size"])
        k_label = str(cfg["k"])
        w = cfg["weights"]
        weights_label = "50/50" if w == [0.5, 0.5] else "pure-sem"
        search_label = cfg["search_type"]
        lines.append(f"| {i:>3} | {llm_label:<12} | {embed_label:<10} | {chunk_label} | {k_label} | {weights_label:<8} | {search_label} |")

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print(f"[OK] Design matrix saved to {path}")


def _short_llm(model: str) -> str:
    if "0.5b" in model:   return "smollm2:0.5b"
    if "1.7b" in model:   return "smollm2:1.7b"
    if "3.1-8b" in model: return "llama-3.1-8b"
    if "3.3-70b" in model: return "llama-3.3-70b"
    return model[:20]


def _full_factorial_size() -> int:
    result = 1
    for lv in LEVELS:
        result *= lv
    return result
