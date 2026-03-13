# evaluation/sweep_config.py
"""
Experiment design for RAG hyperparameter sweep.

Factors (4 active):
  A  llm_model    3 levels: qwen3-8b · ministral-14b · mistral-small-3.2-24b
  B  chunk_size   2 levels: 600 · 1000   (optional 3rd: 800 → 54-run full factorial)
  C  k            2 levels: 2 · 4        (optional 3rd: 3  → 54-run full factorial)
  D  weights      2 levels: pure-semantic · 70/30 hybrid

Fixed (not factors):
  embed_model  = nomic-embed-text-v2-moe  (cosine similarity, Chroma default)
  search_type  = similarity               (cosine, Chroma default)

Full factorial:
  2-level chunk+k: 3 × 2 × 2 × 2 = 24 runs  ← standard Round 1
  3-level chunk+k: 3 × 3 × 3 × 2 = 54 runs  ← optional extended sweep

Usage:
    from sweep_config import build_design, FACTOR_KEYS, save_design_md
    configs = build_design()          # auto: full factorial
    save_design_md(configs, "evaluation/results/doe_design.md")
"""

import os
import json
import itertools
from pathlib import Path

# ---------------------------------------------------------------------------
# Fixed parameters (not swept)
# ---------------------------------------------------------------------------

EMBED_MODEL  = "nomic-embed-text-v2-moe"
SEARCH_TYPE  = "similarity"   # cosine similarity (Chroma default)

# ---------------------------------------------------------------------------
# Parameter grid — maps integer level indices to real values
# ---------------------------------------------------------------------------

PARAM_GRID = {
    "llm_model": [
        "openrouter/qwen/qwen3-8b",
        "openrouter/mistralai/ministral-14b-2512",
        "openrouter/mistralai/mistral-small-3.2-24b-instruct",
    ],
    "chunk_size": [600, 1000],
    "k":          [2, 4],
    "weights": [
        [0.0, 1.0],   # pure semantic (BM25 disabled)
        [0.7, 0.3],   # 70% semantic / 30% BM25
    ],
}

# Factor key order must match LEVELS order below
FACTOR_KEYS = ["llm_model", "chunk_size", "k", "weights"]

# ROUND 1:     LEVELS = [3, 2, 2, 2]  -> 3x2x2x2 = 24 combinations  <- active
# ROUND 1 EXT: LEVELS = [3, 3, 3, 2]  -> 3x3x3x2 = 54 combinations  (uncomment 800+k=3 above)
# SMOKE TEST:  LEVELS = [1, 1, 1, 2]  -> 1x1x1x2 = 2 combinations   (set chunk=[800], k=[4])
LEVELS = [3, 2, 2, 2]   # ROUND 1: 3 × 2 × 2 × 2 = 24 configs

# Human-readable short labels for the design table
FACTOR_LABELS = {
    "llm_model":  ["qwen3-8b", "ministral-14b", "mistral-small-24b"],
    "chunk_size": ["600", "800", "1000"],
    "k":          ["2", "3", "4"],
    "weights":    ["pure-sem", "70/30"],
}


# ---------------------------------------------------------------------------
# Design generation
# ---------------------------------------------------------------------------

def build_design(n_runs: int = None, seed: int = 42) -> list[dict]:
    """
    Generate the experiment design.

    Strategy:
    - If n_runs is None or >= full factorial size: run the full factorial.
      Uses oapackage to verify D-efficiency if installed.
    - If n_runs < full factorial size: fall back to LHS subset (pyDOE3).

    For Round 1 (24 or 54 runs) this always returns the full factorial.
    """
    import numpy as np

    full_factorial_combos = list(itertools.product(*[range(lv) for lv in LEVELS]))
    full_size = len(full_factorial_combos)

    if n_runs is None or n_runs >= full_size:
        matrix = np.array(full_factorial_combos)
        _score_d_efficiency(matrix, full_size)
        return _decode(matrix)

    # Subset via LHS
    try:
        from pyDOE3 import lhs
        np.random.seed(seed)
        criterion = "maximin" if n_runs > 1 else None
        lhs_matrix = lhs(len(FACTOR_KEYS), samples=n_runs, criterion=criterion)
        coded = np.clip(
            np.floor(lhs_matrix * np.array(LEVELS)).astype(int),
            0, np.array(LEVELS) - 1,
        )
        print(f"[OK] LHS design ({n_runs} runs of {full_size} full factorial).")
        return _decode(coded)
    except ImportError:
        print("⚠  pyDOE3 not found — using greedy coverage design.")
        print("   Install with: pip install pyDOE3")
        return _greedy_design(n_runs, seed)


def _score_d_efficiency(matrix, full_size: int) -> None:
    """Print D-efficiency score using oapackage if available, else just confirm full factorial."""
    try:
        import oapackage as oa
        import numpy as np
        # Build coded design matrix as floats for D-optimality scoring
        arr = matrix.astype(float)
        al = oa.array_link(arr.astype(int))
        d_eff = al.Defficiency()
        print(f"[OK] Full factorial ({full_size} runs) - D-efficiency: {d_eff:.4f} (oapackage).")
    except Exception:
        print(f"[OK] Full factorial ({full_size} runs).")


def _decode(coded_matrix) -> list[dict]:
    """Map integer-coded design matrix rows to real parameter dicts."""
    configs = []
    for row in coded_matrix:
        cfg = {key: PARAM_GRID[key][int(val)] for key, val in zip(FACTOR_KEYS, row)}
        # Inject fixed parameters
        cfg["embed_model"] = EMBED_MODEL
        cfg["search_type"] = SEARCH_TYPE
        configs.append(cfg)
    return configs


def _greedy_design(n_runs: int, seed: int = 42) -> list[dict]:
    """Coverage-based fallback: ensures every level appears at least once."""
    import random
    random.seed(seed)

    base_rows = []
    for fi, key in enumerate(FACTOR_KEYS):
        for li in range(LEVELS[fi]):
            row = [random.randint(0, lv - 1) for lv in LEVELS]
            row[fi] = li
            base_rows.append(row)

    seen = set()
    unique = []
    for r in base_rows:
        t = tuple(r)
        if t not in seen:
            seen.add(t)
            unique.append(r)

    all_combos = list(itertools.product(*[range(lv) for lv in LEVELS]))
    random.shuffle(all_combos)
    for combo in all_combos:
        if len(unique) >= n_runs:
            break
        if tuple(combo) not in seen:
            seen.add(tuple(combo))
            unique.append(list(combo))

    import numpy as np
    return _decode(np.array(unique[:n_runs]))


# ---------------------------------------------------------------------------
# Config helpers used by sweep.py
# ---------------------------------------------------------------------------

def config_id(cfg: dict) -> str:
    """Stable string key for a config dict (used for checkpointing)."""
    return json.dumps(cfg, sort_keys=True)


def chroma_dir(cfg: dict, base: str = None) -> str:
    """
    Unique persist directory for a chunk_size.
    Always uses nomic embedding — only chunk_size varies between dirs.
    Reuses the production chroma_db for chunk=800 (already embedded).
    """
    root = os.path.dirname(os.path.dirname(__file__))
    chunk = cfg["chunk_size"]
    if chunk == 800:
        return os.path.join(root, "chroma_db")
    if base is None:
        base = os.path.join(root, "chroma_db_sweep")
    return os.path.join(base, f"nomic_c{chunk}")


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

    full_size = 1
    for lv in LEVELS:
        full_size *= lv

    lines = [
        "# Experiment Design Matrix\n",
        f"**Runs:** {len(configs)}  |  **Factors:** {len(FACTOR_KEYS)}  |  "
        f"**Full factorial:** {full_size} runs\n",
        f"**Fixed:** embed=nomic  |  search=cosine-similarity\n",
        "",
        "| Run | LLM | Chunk | k | Weights |",
        "|-----|-----|-------|---|---------|",
    ]

    for i, cfg in enumerate(configs, 1):
        llm_label    = _short_llm(cfg["llm_model"])
        chunk_label  = str(cfg["chunk_size"])
        k_label      = str(cfg["k"])
        w            = cfg["weights"]
        weights_label = "pure-sem" if w == [0.0, 1.0] else "70/30"
        lines.append(f"| {i:>3} | {llm_label:<14} | {chunk_label} | {k_label} | {weights_label} |")

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print(f"[OK] Design matrix saved to {path}")


def _short_llm(model: str) -> str:
    if "qwen3-8b"   in model: return "qwen3-8b"
    if "ministral"  in model: return "ministral-14b"
    if "mistral-small" in model: return "mistral-small-24b"
    if "3.3-70b"    in model: return "llama-3.3-70b"
    return model[:20]


def _full_factorial_size() -> int:
    result = 1
    for lv in LEVELS:
        result *= lv
    return result
