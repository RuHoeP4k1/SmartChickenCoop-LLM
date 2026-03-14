# evaluation/plot_results.py
"""
Generate presentation-ready plots from RAG sweep CSV results.

Outputs (saved to evaluation/results/plots/):
  1. fig1_llm_comparison.png   — grouped bar chart: 3 LLMs × 5 metrics
  2. fig2_retrieval_heatmap.png — 2×2 heatmap: chunk × k on contextual recall
  3. fig3_radar.png             — radar chart: best config vs baseline
  4. fig4_top5_table.png        — top-5 configs by overall weighted score

Usage:
    python evaluation/plot_results.py
    python evaluation/plot_results.py --input evaluation/results/sweep_XXX.csv
"""

import os
import sys
import glob
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import matplotlib.gridspec as gridspec

_HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(_HERE, "results")
PLOTS_DIR   = os.path.join(RESULTS_DIR, "plots")

# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------

PALETTE   = ["#4C72B0", "#DD8452", "#55A868"]   # blue / orange / green
LLM_ORDER = ["qwen3-8b", "ministral-14b", "mistral-small-24b"]
LLM_SHORT = {"qwen3-8b": "Qwen3-8B", "ministral-14b": "Ministral-14B",
             "mistral-small-24b": "Mistral-Small-24B"}

METRICS = ["actionability", "correctness", "faithfulness",
           "answer_relevancy", "contextual_recall"]
METRIC_LABELS = {
    "actionability":    "Actionability",
    "correctness":      "Correctness",
    "faithfulness":     "Faithfulness",
    "answer_relevancy": "Answer\nRelevancy",
    "contextual_recall":"Contextual\nRecall",
}

WEIGHTS_OVERALL = {
    "actionability":    0.25,
    "correctness":      0.25,
    "answer_relevancy": 0.20,
    "faithfulness":     0.15,
    "contextual_recall":0.15,
}

plt.rcParams.update({
    "font.family":  "DejaVu Sans",
    "font.size":    11,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "figure.dpi":   150,
})


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def load_csv(input_path: str = None) -> pd.DataFrame:
    if input_path is None:
        files = sorted(glob.glob(os.path.join(RESULTS_DIR, "sweep_*.csv")))
        if not files:
            raise FileNotFoundError(f"No sweep CSV in {RESULTS_DIR}")
        input_path = files[-1]
    print(f"Loading: {input_path}")
    return pd.read_csv(input_path)


def overall_score(row: pd.Series) -> float:
    total_w, total_s = 0.0, 0.0
    for m, w in WEIGHTS_OVERALL.items():
        v = row.get(m, -1)
        if v is not None and not np.isnan(float(v)) and float(v) >= 0:
            total_s += float(v) * w
            total_w += w
    return total_s / total_w if total_w > 0 else np.nan


def add_overall(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["overall"] = df.apply(overall_score, axis=1)
    return df


def valid_mean(series: pd.Series) -> float:
    return series[series >= 0].mean()


def valid_sem(series: pd.Series) -> float:
    s = series[series >= 0]
    return s.sem() if len(s) > 1 else 0.0


# ---------------------------------------------------------------------------
# Figure 1 — LLM comparison bar chart
# ---------------------------------------------------------------------------

def fig1_llm_comparison(df: pd.DataFrame, out_dir: str) -> None:
    fig, ax = plt.subplots(figsize=(11, 5))

    n_metrics = len(METRICS)
    n_llms    = len(LLM_ORDER)
    x         = np.arange(n_metrics)
    bar_w     = 0.25
    offsets   = np.linspace(-(n_llms - 1) / 2 * bar_w,
                             (n_llms - 1) / 2 * bar_w, n_llms)

    for idx, (llm, offset, color) in enumerate(zip(LLM_ORDER, offsets, PALETTE)):
        sub = df[df["llm"] == llm]
        means = [valid_mean(sub[m]) for m in METRICS]
        sems  = [valid_sem(sub[m])  for m in METRICS]
        bars  = ax.bar(x + offset, means, bar_w,
                       label=LLM_SHORT[llm], color=color, alpha=0.85,
                       yerr=sems, capsize=3, error_kw={"elinewidth": 1.2})

    ax.set_xticks(x)
    ax.set_xticklabels([METRIC_LABELS[m] for m in METRICS], fontsize=10)
    ax.set_ylabel("Mean Score (0–1)")
    ax.set_ylim(0.6, 1.05)
    ax.set_title("LLM Model Comparison across Evaluation Metrics", fontsize=13, pad=12)
    ax.legend(loc="lower right", framealpha=0.9)
    ax.axhline(1.0, color="grey", linewidth=0.5, linestyle="--")

    fig.tight_layout()
    path = os.path.join(out_dir, "fig1_llm_comparison.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ---------------------------------------------------------------------------
# Figure 2 — Retrieval heatmap (chunk × k → contextual recall)
# ---------------------------------------------------------------------------

def fig2_retrieval_heatmap(df: pd.DataFrame, out_dir: str) -> None:
    chunks = sorted(df["chunk"].unique())
    ks     = sorted(df["k"].unique())

    data = np.zeros((len(chunks), len(ks)))
    for ci, c in enumerate(chunks):
        for ki, k in enumerate(ks):
            sub = df[(df["chunk"] == c) & (df["k"] == k)]["contextual_recall"]
            data[ci, ki] = valid_mean(sub)

    fig, ax = plt.subplots(figsize=(5.5, 4))
    im = ax.imshow(data, cmap="RdYlGn", vmin=0.6, vmax=1.0, aspect="auto")

    ax.set_xticks(range(len(ks)))
    ax.set_yticks(range(len(chunks)))
    ax.set_xticklabels([f"k = {k}" for k in ks], fontsize=11)
    ax.set_yticklabels([f"chunk = {c}" for c in chunks], fontsize=11)
    ax.set_title("Contextual Recall\nby Chunk Size × k", fontsize=12, pad=10)

    for ci in range(len(chunks)):
        for ki in range(len(ks)):
            ax.text(ki, ci, f"{data[ci, ki]:.3f}",
                    ha="center", va="center", fontsize=13, fontweight="bold",
                    color="black")

    cbar = fig.colorbar(im, ax=ax, shrink=0.85)
    cbar.set_label("Mean Score", fontsize=9)

    fig.tight_layout()
    path = os.path.join(out_dir, "fig2_retrieval_heatmap.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ---------------------------------------------------------------------------
# Figure 3 — Radar chart: best config vs baseline
# ---------------------------------------------------------------------------

def fig3_radar(df: pd.DataFrame, out_dir: str) -> None:
    best_mask = (
        (df["llm"] == "ministral-14b") &
        (df["chunk"] == 1000) &
        (df["k"] == 4) &
        (df["weights"] == "70/30")
    )
    base_mask = (
        (df["llm"] == "qwen3-8b") &
        (df["chunk"] == 600) &
        (df["k"] == 2) &
        (df["weights"] == "pure-sem")
    )

    best_vals = [valid_mean(df[best_mask][m]) for m in METRICS]
    base_vals = [valid_mean(df[base_mask][m]) for m in METRICS]

    labels = [METRIC_LABELS[m].replace("\n", " ") for m in METRICS]
    N      = len(labels)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    best_vals += best_vals[:1]
    base_vals += base_vals[:1]

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, polar=True)

    ax.plot(angles, best_vals, color=PALETTE[1], linewidth=2.2,
            label="Best config\n(Ministral-14B | chunk=1000 | k=4 | 70/30)")
    ax.fill(angles, best_vals, color=PALETTE[1], alpha=0.20)
    ax.plot(angles, base_vals, color=PALETTE[0], linewidth=2.2,
            linestyle="--",
            label="Baseline config\n(Qwen3-8B | chunk=600 | k=2 | pure-sem)")
    ax.fill(angles, base_vals, color=PALETTE[0], alpha=0.12)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0.25", "0.50", "0.75", "1.00"], fontsize=8, color="grey")
    ax.grid(color="grey", linestyle="--", linewidth=0.5, alpha=0.6)

    ax.set_title("Best vs Baseline Configuration\nacross All Metrics",
                 fontsize=13, pad=25)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.18),
              framealpha=0.9, fontsize=9, ncol=2)

    fig.subplots_adjust(bottom=0.18)
    path = os.path.join(out_dir, "fig3_radar.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ---------------------------------------------------------------------------
# Figure 4 — Top-5 configs table
# ---------------------------------------------------------------------------

def fig4_top5_table(df: pd.DataFrame, out_dir: str) -> None:
    df_overall = add_overall(df)

    # Average per config
    config_cols = ["llm", "chunk", "k", "weights"]
    agg = (
        df_overall.groupby(config_cols)["overall"]
        .mean()
        .reset_index()
        .sort_values("overall", ascending=False)
        .head(5)
        .reset_index(drop=True)
    )
    agg.insert(0, "Rank", range(1, len(agg) + 1))
    agg["overall"] = agg["overall"].round(4)
    agg.columns = ["Rank", "LLM", "Chunk", "k", "Weights", "Overall Score"]
    agg["LLM"] = agg["LLM"].map(LLM_SHORT).fillna(agg["LLM"])

    fig, ax = plt.subplots(figsize=(9, 2.8))
    ax.axis("off")

    col_widths = [0.06, 0.28, 0.10, 0.06, 0.14, 0.18]
    table = ax.table(
        cellText=agg.values,
        colLabels=agg.columns,
        cellLoc="center",
        loc="center",
        colWidths=col_widths,
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 1.8)

    # Header style
    for j in range(len(agg.columns)):
        cell = table[0, j]
        cell.set_facecolor("#2C5F8A")
        cell.set_text_props(color="white", fontweight="bold")

    # Row alternating colours + highlight top row
    for i in range(1, len(agg) + 1):
        bg = "#FFF3CD" if i == 1 else ("#F2F2F2" if i % 2 == 0 else "white")
        for j in range(len(agg.columns)):
            table[i, j].set_facecolor(bg)

    ax.set_title("Top 5 Configurations by Overall Weighted Score",
                 fontsize=12, pad=10, loc="left")

    fig.tight_layout()
    path = os.path.join(out_dir, "fig4_top5_table.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", help="Path to sweep CSV")
    args = parser.parse_args()

    df = load_csv(args.input)
    Path(PLOTS_DIR).mkdir(parents=True, exist_ok=True)

    print("\nGenerating figures...")
    fig1_llm_comparison(df, PLOTS_DIR)
    fig2_retrieval_heatmap(df, PLOTS_DIR)
    fig3_radar(df, PLOTS_DIR)
    fig4_top5_table(df, PLOTS_DIR)

    print(f"\nAll figures saved to: {PLOTS_DIR}")


if __name__ == "__main__":
    main()
