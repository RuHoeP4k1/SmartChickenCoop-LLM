# evaluation/sweep_analysis.py
"""
Analyse RAG sweep results.

Outputs:
  A. Main effects table   — avg score per level per factor
  B. One-way ANOVA        — F-stat + p-value per factor (scipy required)
  C. Two-way interactions — text heatmap for significant factor pairs
  D. Ranked config table  — top 10 by combined score
  E. Markdown report      — saved to evaluation/results/round1_analysis.md

Usage:
    python evaluation/sweep_analysis.py                             # latest checkpoint or latest sweep_*.json
    python evaluation/sweep_analysis.py --input results/sweep_X.json
    python evaluation/sweep_analysis.py --metric actionability      # focus on one metric
"""

import os
import sys
import json
import argparse
import glob
from pathlib import Path
from collections import defaultdict

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _HERE)

from sweep_config import FACTOR_KEYS, PARAM_GRID, FACTOR_LABELS

RESULTS_DIR = os.path.join(_HERE, "results")
REPORT_PATH = os.path.join(RESULTS_DIR, "round1_analysis.md")


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_results(input_path: str = None) -> dict:
    """Load sweep results JSON. Auto-detects latest file if not specified."""
    if input_path is None:
        # Prefer checkpoint (in-progress run), then most recent sweep file
        checkpoint = os.path.join(RESULTS_DIR, "sweep_checkpoint.json")
        if os.path.exists(checkpoint):
            print(f"Loading checkpoint: {checkpoint}")
            raw = json.load(open(checkpoint, encoding="utf-8"))
            # Checkpoint is just a list; wrap it
            return {"results": raw, "timestamp": "checkpoint", "judge_model": "unknown", "n_questions": "?"}

        pattern = os.path.join(RESULTS_DIR, "sweep_*.json")
        files = sorted(glob.glob(pattern))
        if not files:
            raise FileNotFoundError(f"No sweep results found in {RESULTS_DIR}. Run sweep.py first.")
        input_path = files[-1]
        print(f"Loading: {input_path}")

    with open(input_path, encoding="utf-8") as f:
        return json.load(f)


def extract_results(data: dict) -> list:
    """Extract the list of per-config results regardless of file format."""
    if isinstance(data, list):
        return data
    return data.get("results", data)


# ---------------------------------------------------------------------------
# Score extraction helpers
# ---------------------------------------------------------------------------

def get_score(result: dict, metric: str) -> float:
    """Get the average score for a metric from a result dict. Returns -1 if metric failed entirely."""
    if metric == "combined":
        v = result.get("avg_combined")
        if v is not None:
            return v
        a = result.get("avg_actionability", -1)
        c = result.get("avg_correctness", -1)
        vals = [x for x in [a, c] if x >= 0]
        return sum(vals) / len(vals) if vals else -1.0
    return result.get(f"avg_{metric}", -1.0)


def level_label(factor: str, value) -> str:
    """Human-readable label for a factor level value."""
    grid = PARAM_GRID[factor]
    try:
        idx = grid.index(value)
        labels = FACTOR_LABELS.get(factor, [])
        return labels[idx] if idx < len(labels) else str(value)
    except ValueError:
        return str(value)


# ---------------------------------------------------------------------------
# A. Main effects table
# ---------------------------------------------------------------------------

def main_effects(results: list, metric: str = "combined") -> dict:
    """
    Compute average score per level per factor.
    Returns: {factor: {level_label: (mean, count)}}
    """
    buckets = {fk: defaultdict(list) for fk in FACTOR_KEYS}

    for r in results:
        cfg   = r["config"]
        score = get_score(r, metric)
        if score < 0:
            continue   # metric failed entirely for this config — exclude from effects
        for fk in FACTOR_KEYS:
            lbl = level_label(fk, cfg[fk])
            buckets[fk][lbl].append(score)

    effects = {}
    for fk in FACTOR_KEYS:
        effects[fk] = {
            lbl: (sum(vals) / len(vals), len(vals))
            for lbl, vals in buckets[fk].items()
        }
    return effects


def print_main_effects(effects: dict, metric: str = "combined") -> str:
    lines = [
        f"\n{'='*70}",
        f"A. MAIN EFFECTS  (metric: {metric})",
        f"{'='*70}",
        f"{'Factor':<14}  {'Level':<14}  {'Avg Score':>10}  {'N':>5}  {'Range':>8}",
        f"{'-'*70}",
    ]

    factor_ranges = {}
    for fk, levels in effects.items():
        vals = [v for v, _ in levels.values()]
        factor_ranges[fk] = max(vals) - min(vals)

    # Sort factors by descending range (most impactful first)
    sorted_factors = sorted(FACTOR_KEYS, key=lambda fk: factor_ranges[fk], reverse=True)

    for fk in sorted_factors:
        levels = effects[fk]
        # Sort levels by score descending
        sorted_levels = sorted(levels.items(), key=lambda x: x[1][0], reverse=True)
        rng = factor_ranges[fk]
        for i, (lbl, (mean, cnt)) in enumerate(sorted_levels):
            factor_col = fk if i == 0 else ""
            range_col  = f"{rng:+.4f}" if i == 0 else ""
            lines.append(f"{factor_col:<14}  {lbl:<14}  {mean:>10.4f}  {cnt:>5}  {range_col:>8}")
        lines.append("")

    text = "\n".join(lines)
    print(text)
    return text


# ---------------------------------------------------------------------------
# B. One-way ANOVA
# ---------------------------------------------------------------------------

def run_anova(results: list, metric: str = "combined") -> list:
    """
    One-way ANOVA for each factor.
    Returns list of (factor, F_stat, p_value, significant) sorted by F descending.
    """
    try:
        from scipy import stats
    except ImportError:
        print("⚠  scipy not installed — skipping ANOVA. Run: pip install scipy")
        return []

    anova_results = []
    for fk in FACTOR_KEYS:
        buckets = defaultdict(list)
        for r in results:
            score = get_score(r, metric)
            if score < 0:
                continue
            lbl = level_label(fk, r["config"][fk])
            buckets[lbl].append(score)

        groups = list(buckets.values())
        if len(groups) < 2 or any(len(g) < 2 for g in groups):
            anova_results.append((fk, float("nan"), float("nan"), False))
            continue

        f_stat, p_val = stats.f_oneway(*groups)
        anova_results.append((fk, f_stat, p_val, p_val < 0.05))

    return sorted(anova_results, key=lambda x: (0 if x[1] != x[1] else -x[1]))


def print_anova(anova_results: list) -> str:
    lines = [
        f"\n{'='*60}",
        "B. ONE-WAY ANOVA  (H₀: all levels produce the same score)",
        f"{'='*60}",
        f"{'Factor':<14}  {'F-stat':>8}  {'p-value':>9}  {'Significant':>12}",
        f"{'-'*60}",
    ]
    for fk, f_stat, p_val, sig in anova_results:
        f_str = f"{f_stat:.2f}" if f_stat == f_stat else "  n/a"
        p_str = f"{p_val:.4f}" if p_val == p_val else "   n/a"
        sig_str = "*** SIGNIFICANT" if sig else ""
        lines.append(f"{fk:<14}  {f_str:>8}  {p_str:>9}  {sig_str}")
    lines.append(f"{'='*60}")
    lines.append("Threshold: p < 0.05")
    text = "\n".join(lines)
    print(text)
    return text


# ---------------------------------------------------------------------------
# C. Two-way interaction tables
# ---------------------------------------------------------------------------

def interaction_table(results: list, factor_a: str, factor_b: str, metric: str = "combined") -> str:
    """Build a text heatmap for two factors."""
    levels_a = list(FACTOR_LABELS[factor_a])
    levels_b = list(FACTOR_LABELS[factor_b])

    buckets = defaultdict(list)
    for r in results:
        score = get_score(r, metric)
        if score < 0:
            continue
        la = level_label(factor_a, r["config"][factor_a])
        lb = level_label(factor_b, r["config"][factor_b])
        buckets[(la, lb)].append(score)

    # Header
    col_w = 10
    header = f"  {factor_a} \\ {factor_b:<{col_w}}"
    for lb in levels_b:
        header += f"  {lb:>{col_w}}"
    lines = [header, "-" * len(header)]

    for la in levels_a:
        row = f"  {la:<{col_w + len(factor_a) + 3 - col_w}}"
        for lb in levels_b:
            vals = buckets.get((la, lb), [])
            cell = f"{sum(vals)/len(vals):.4f}" if vals else "  n/a  "
            row += f"  {cell:>{col_w}}"
        lines.append(row)

    return "\n".join(lines)


def print_interactions(results: list, anova_results: list, metric: str = "combined") -> str:
    sig_factors = [fk for fk, _, _, sig in anova_results if sig]
    if len(sig_factors) < 2:
        msg = "\nC. TWO-WAY INTERACTIONS\n  (fewer than 2 significant factors — skipping)\n"
        print(msg)
        return msg

    lines = [f"\n{'='*60}", "C. TWO-WAY INTERACTIONS", f"{'='*60}"]
    import itertools
    for fa, fb in itertools.combinations(sig_factors, 2):
        lines.append(f"\n  {fa}  ×  {fb}  (mean {metric} score):")
        lines.append(interaction_table(results, fa, fb, metric))

    text = "\n".join(lines)
    print(text)
    return text


# ---------------------------------------------------------------------------
# D. Ranked config table
# ---------------------------------------------------------------------------

def print_ranked_table(results: list, top_n: int = 10) -> str:
    sorted_r = sorted(results, key=lambda r: get_score(r, "combined"), reverse=True)

    lines = [
        f"\n{'='*80}",
        f"D. TOP {top_n} CONFIGURATIONS  (by avg combined score)",
        f"{'='*80}",
        f"{'Rank':<5} {'LLM':<15} {'Chunk':<6} {'k':<3} {'Weights':<10} "
        f"{'Action':>8} {'Correct':>8} {'Combined':>9}",
        "-" * 80,
    ]

    for rank, r in enumerate(sorted_r[:top_n], 1):
        cfg     = r["config"]
        llm     = _short_llm(cfg["llm_model"])
        chunk   = str(cfg["chunk_size"])
        k       = str(cfg["k"])
        w       = cfg["weights"]
        weights = "pure-sem" if w == [0.0, 1.0] else "70/30"
        lines.append(
            f"{rank:<5} {llm:<15} {chunk:<6} {k:<3} {weights:<10} "
            f"{r['avg_actionability']:>8.4f} {r['avg_correctness']:>8.4f} {get_score(r, 'combined'):>9.4f}"
        )

    lines.append("=" * 80)
    text = "\n".join(lines)
    print(text)
    return text


def _short_llm(model: str) -> str:
    if "1.7b"    in model: return "smollm2:1.7b"
    if "3.1-8b"  in model: return "llama-3.1-8b"
    if "3.2-24b" in model: return "mistral-3.2-24b"
    if "3.3-70b" in model: return "llama-3.3-70b"
    return model[:14]


# ---------------------------------------------------------------------------
# CSV exports
# ---------------------------------------------------------------------------

METRICS = [
    "actionability", "correctness", "faithfulness",
    "answer_relevancy", "contextual_precision", "contextual_recall", "contextual_relevancy",
]


def export_long_csv(results: list, path: str) -> None:
    """
    Long-format CSV: one row per question x config.
    Use this for mixed-model analysis in JMP/SAS (question + category as random effects).

    Columns: question_id, category, llm_model, chunk_size, k, weights,
             actionability, correctness, faithfulness, answer_relevancy,
             contextual_precision, contextual_recall, contextual_relevancy
    """
    import csv
    Path(path).parent.mkdir(parents=True, exist_ok=True)

    fieldnames = ["question_id", "category", "llm_model", "chunk_size", "k", "weights"] + METRICS + ["composite"]

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for r in results:
            cfg = r["config"]
            w   = cfg["weights"]
            weights_label = "pure-sem" if w == [0.0, 1.0] else "70/30"

            for qi, q in enumerate(r.get("per_question", []), 1):
                row = {
                    "question_id": qi,
                    "category":    q.get("category", "unknown"),
                    "llm_model":   _short_llm(cfg["llm_model"]),
                    "chunk_size":  cfg["chunk_size"],
                    "k":           cfg["k"],
                    "weights":     weights_label,
                }
                scores = []
                for m in METRICS:
                    v = q.get(m, "")
                    row[m] = v
                    if isinstance(v, (int, float)) and v >= 0:
                        scores.append(v)
                row["composite"] = round(sum(scores) / len(scores), 4) if scores else ""
                writer.writerow(row)

    print(f"[OK] Long-format CSV saved -> {path}  (for JMP/SAS mixed models)")


def export_averaged_csv(results: list, path: str) -> None:
    """
    Averaged CSV: one row per config (averages across all questions).
    Use this for quick sanity checks and simple ANOVA.

    Columns: llm_model, chunk_size, k, weights, avg_actionability, avg_correctness,
             avg_faithfulness, avg_answer_relevancy, avg_contextual_precision,
             avg_contextual_recall, avg_contextual_relevancy, avg_combined
    """
    import csv
    Path(path).parent.mkdir(parents=True, exist_ok=True)

    avg_cols   = [f"avg_{m}" for m in METRICS] + ["avg_combined", "composite"]
    fieldnames = ["llm_model", "chunk_size", "k", "weights"] + avg_cols

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for r in results:
            cfg = r["config"]
            w   = cfg["weights"]
            weights_label = "pure-sem" if w == [0.0, 1.0] else "70/30"
            row = {
                "llm_model":  _short_llm(cfg["llm_model"]),
                "chunk_size": cfg["chunk_size"],
                "k":          cfg["k"],
                "weights":    weights_label,
            }
            for col in [f"avg_{m}" for m in METRICS] + ["avg_combined"]:
                row[col] = r.get(col, "")
            # composite: mean of all 7 avg metrics (excluding -1 sentinels)
            vals = [r.get(f"avg_{m}", -1) for m in METRICS]
            valid = [v for v in vals if isinstance(v, (int, float)) and v >= 0]
            row["composite"] = round(sum(valid) / len(valid), 4) if valid else ""
            writer.writerow(row)

    print(f"[OK] Averaged CSV saved    -> {path}  (for quick sanity checks)")


# ---------------------------------------------------------------------------
# E. Markdown report
# ---------------------------------------------------------------------------

def save_markdown_report(
    data: dict,
    results: list,
    effects_text: str,
    anova_text: str,
    interaction_text: str,
    ranked_text: str,
    metric: str,
) -> None:
    Path(RESULTS_DIR).mkdir(parents=True, exist_ok=True)

    winner = sorted(results, key=lambda r: get_score(r, "combined"), reverse=True)[0]
    wcfg   = winner["config"]

    lines = [
        "# Round 1 Sweep Analysis\n",
        f"**Timestamp:** {data.get('timestamp', 'unknown')}  ",
        f"**Judge:** {data.get('judge_model', 'unknown')}  ",
        f"**Questions per config:** {data.get('n_questions', '?')}  ",
        f"**Configs completed:** {len(results)}  \n",
        "---\n",
        "## Winner Config\n",
        f"| Factor | Value |",
        f"|--------|-------|",
        f"| LLM | `{wcfg['llm_model']}` |",
        f"| Chunk size | {wcfg['chunk_size']} |",
        f"| k | {wcfg['k']} |",
        f"| Weights | {'pure-sem' if wcfg['weights'] == [0.0, 1.0] else '70/30'} |",
        f"\n**Combined score:** {get_score(winner, 'combined'):.4f}  ",
        f"**Actionability:** {winner['avg_actionability']:.4f}  ",
        f"**Correctness:** {winner['avg_correctness']:.4f}\n",
        "---\n",
        "## A. Main Effects\n",
        "```",
        effects_text,
        "```\n",
        "---\n",
        "## B. ANOVA\n",
        "```",
        anova_text,
        "```\n",
        "---\n",
        "## C. Two-Way Interactions\n",
        "```",
        interaction_text,
        "```\n",
        "---\n",
        "## D. Top 10 Configurations\n",
        "```",
        ranked_text,
        "```\n",
        "---\n",
        "## Prompt Review Checklist\n",
        "- [ ] Read answers for top 3 configs — what does a good answer look like?",
        "- [ ] Read answers for bottom 3 configs — what failure modes appear?",
        "- [ ] Do small models (0.5b) fail on retrieval or generation?",
        "- [ ] Are larger models over-verbose? Check length calibration.",
        "- [ ] Any prompt phrasing that systematically confuses models?",
        "- [ ] Update `prompts.py` if needed. Tag commit: `prompt-v2`.",
        "\n---\n",
        "## Round 2 Plan\n",
        "1. Fix low-impact factors (see ANOVA) at their best level.",
        "2. Run full factorial on top 2–3 significant factors.",
        "3. Re-run winner config with new prompt to measure prompt impact.",
        "4. Merge human ranking CSV if available.",
    ]

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"\n[OK] Markdown report saved → {REPORT_PATH}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Analyse RAG sweep results")
    parser.add_argument("--input",   default=None,       help="Path to sweep JSON (default: latest)")
    parser.add_argument("--metric",  default="combined",
                        choices=["actionability", "correctness", "combined"],
                        help="Primary metric for ranking (default: combined)")
    parser.add_argument("--top",     type=int, default=10, help="Rows in ranked table")
    args = parser.parse_args()

    data    = load_results(args.input)
    results = extract_results(data)
    # Resolve actual input path for CSV naming
    if args.input is None:
        _files = sorted(glob.glob(os.path.join(RESULTS_DIR, "sweep_*.json")))
        args.input = _files[-1] if _files else "sweep"

    if not results:
        print("No results found.")
        return

    print(f"\nLoaded {len(results)} configs. Analysing metric: {args.metric}\n")

    effects      = main_effects(results, args.metric)
    effects_text = print_main_effects(effects, args.metric)

    anova_results = run_anova(results, args.metric)
    anova_text    = print_anova(anova_results) if anova_results else "(scipy not installed)"

    interaction_text = print_interactions(results, anova_results, args.metric)
    ranked_text      = print_ranked_table(results, top_n=args.top)

    save_markdown_report(
        data, results,
        effects_text, anova_text, interaction_text, ranked_text,
        args.metric,
    )

    # CSV exports
    stem = os.path.splitext(os.path.basename(args.input or "sweep"))[0]
    export_long_csv(results,     os.path.join(RESULTS_DIR, f"{stem}_long.csv"))
    export_averaged_csv(results, os.path.join(RESULTS_DIR, f"{stem}_averaged.csv"))


if __name__ == "__main__":
    main()
