# evaluation/sweep_mixed_model.py
"""
Linear Mixed Effects analysis of RAG sweep results.

Fits one LME model per metric/composite with a random intercept per question
(q_num) to account for repeated-measures structure (same questions across all
configs).  Outputs:

  A. Fixed effects table per metric          (Type III Wald tests)
  B. Pairwise LLM contrasts                  (Bonferroni-corrected)
  C. Group A — LLM answer quality composite
  D. Group B — RAG retrieval quality composite
  E. Overall weighted score composite
  F. Bad question detection via BLUPs

Usage:
    python evaluation/sweep_mixed_model.py
    python evaluation/sweep_mixed_model.py --input evaluation/results/sweep_XXX.csv
"""

import os
import sys
import glob
import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
_EVAL = os.path.dirname(_HERE)
_ROOT = os.path.dirname(_EVAL)
sys.path.insert(0, _EVAL)

RESULTS_DIR = os.path.join(_EVAL, "results")
REPORT_PATH = os.path.join(RESULTS_DIR, "reports", "mixed_model_analysis.md")

# ---------------------------------------------------------------------------
# Metric definitions
# ---------------------------------------------------------------------------

# All scored columns in the CSV
ALL_METRICS = ["actionability", "correctness", "answer_relevancy",
               "faithfulness", "contextual_recall"]

# Group A: LLM answer quality
GROUP_A = ["actionability", "correctness", "answer_relevancy"]
GROUP_A_LABEL = "LLM Answer Quality (A)"

# Group B: RAG retrieval quality
GROUP_B = ["contextual_recall", "faithfulness"]
GROUP_B_LABEL = "RAG Retrieval Quality (B)"

# Overall weighted score weights (must sum to 1.0)
WEIGHTS = {
    "actionability":    0.25,
    "correctness":      0.25,
    "answer_relevancy": 0.20,
    "faithfulness":     0.15,
    "contextual_recall":0.15,
}

# Reference levels for factor coding
REF = {
    "llm":     "qwen3-8b",
    "chunk":   600,
    "k":       2,
    "weights": "pure-sem",
}

# Bad-question threshold: BLUPs below mean - N_SD * std are flagged
BAD_Q_SD_THRESHOLD = 1.5


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_csv(input_path: str = None) -> pd.DataFrame:
    if input_path is None:
        pattern = os.path.join(RESULTS_DIR, "sweep_*.csv")
        files = sorted(glob.glob(pattern))
        if not files:
            raise FileNotFoundError(f"No sweep CSV found in {RESULTS_DIR}")
        input_path = files[-1]
    print(f"Loading: {input_path}\n")
    df = pd.read_csv(input_path)
    return df


def load_questions() -> dict:
    """Return {q_num: question_text} from evaluation_data.py TEST_CASES."""
    try:
        from evaluation_data import TEST_CASES
        return {i + 1: tc["question"] for i, tc in enumerate(TEST_CASES)}
    except ImportError:
        return {}


# ---------------------------------------------------------------------------
# Composite score helpers
# ---------------------------------------------------------------------------

def weighted_row(row: pd.Series, metric_cols: list, weights: dict = None) -> float:
    """
    Compute a weighted average for one row, ignoring -1 (failed) metrics.
    If weights is None, uses equal weights.
    """
    vals, ws = [], []
    for m in metric_cols:
        v = row.get(m, -1)
        if v is not None and not np.isnan(float(v)) and float(v) >= 0:
            w = weights[m] if weights else 1.0
            vals.append(float(v))
            ws.append(w)
    if not vals:
        return np.nan
    total_w = sum(ws)
    return sum(v * w for v, w in zip(vals, ws)) / total_w


def add_composites(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["composite_A"] = df.apply(
        lambda r: weighted_row(r, GROUP_A), axis=1)
    df["composite_B"] = df.apply(
        lambda r: weighted_row(r, GROUP_B), axis=1)
    df["composite_overall"] = df.apply(
        lambda r: weighted_row(r, ALL_METRICS, WEIGHTS), axis=1)
    return df


# ---------------------------------------------------------------------------
# Mixed model fitting
# ---------------------------------------------------------------------------

def fit_lme(df: pd.DataFrame, outcome: str):
    """
    Fit  outcome ~ C(llm) + C(chunk) + C(k) + C(weights)  with random
    intercept on q_num.  Returns the fitted result or None on failure.
    """
    try:
        import statsmodels.formula.api as smf
    except ImportError:
        raise ImportError("statsmodels is required: pip install statsmodels")

    sub = df[["q_num", "llm", "chunk", "k", "weights", outcome]].dropna()
    sub = sub[sub[outcome] >= 0].copy()

    if len(sub) < 20:
        print(f"  ⚠  Too few valid rows for {outcome} ({len(sub)}) — skipping model")
        return None

    # Set reference levels via category dtype
    sub["llm"]     = pd.Categorical(sub["llm"],     categories=_ordered_cats(sub["llm"],     REF["llm"]))
    sub["weights"] = pd.Categorical(sub["weights"],  categories=_ordered_cats(sub["weights"],  REF["weights"]))
    sub["chunk"]   = pd.Categorical(sub["chunk"].astype(str),
                                    categories=_ordered_cats(sub["chunk"].astype(str), str(REF["chunk"])))
    sub["k"]       = pd.Categorical(sub["k"].astype(str),
                                    categories=_ordered_cats(sub["k"].astype(str), str(REF["k"])))

    formula = f"{outcome} ~ C(llm) + C(chunk) + C(k) + C(weights)"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = smf.mixedlm(formula, data=sub, groups=sub["q_num"])
        # Try powell first (handles boundary better), fall back to lbfgs
        try:
            result = model.fit(reml=True, method="powell")
        except Exception:
            result = model.fit(reml=True, method="lbfgs")
    sigma_q = result.cov_re.values[0][0]
    print(f"  σ²_q = {sigma_q:.6f}  (n={int(result.nobs)})")
    return result


def _ordered_cats(series, ref):
    """Categorical levels with ref first."""
    others = sorted(set(series.astype(str).unique()) - {str(ref)})
    return [str(ref)] + others


# ---------------------------------------------------------------------------
# Fixed effects table
# ---------------------------------------------------------------------------

def fixed_effects_table(result, title: str) -> list[str]:
    """Return markdown lines for the fixed effects table."""
    fe = result.fe_params
    ci = result.conf_int()
    pvals = result.pvalues

    sigma_q = result.cov_re.values[0][0]
    lines = [f"### {title}", ""]
    lines.append(f"Random effects variance (σ²_q): {sigma_q:.6f}")
    lines.append(f"N observations: {int(result.nobs)}")
    lines.append("")
    lines.append("| Parameter | Coef | SE | z | p | 95% CI | Sig |")
    lines.append("|-----------|------|----|---|---|--------|-----|")

    for param in fe.index:
        if param == "Intercept":
            continue
        coef = fe[param]
        se   = result.bse[param]
        z    = result.tvalues[param]
        p    = pvals[param]
        lo   = ci.loc[param, 0]
        hi   = ci.loc[param, 1]
        sig  = "**" if p < 0.01 else ("*" if p < 0.05 else "")
        label = _clean_param(param)
        p_str = "< .001" if p < 0.001 else f"{p:.3f}"
        lines.append(f"| {label} | {coef:+.4f} | {se:.4f} | {z:+.2f} | {p_str} | [{lo:+.4f}, {hi:+.4f}] | {sig} |")

    lines.append("")
    return lines


def _clean_param(s: str) -> str:
    """Make statsmodels parameter names readable."""
    s = s.replace("C(llm)[T.", "llm=").replace("C(chunk)[T.", "chunk=") \
         .replace("C(k)[T.", "k=").replace("C(weights)[T.", "weights=") \
         .rstrip("]")
    return s


# ---------------------------------------------------------------------------
# Pairwise LLM contrasts
# ---------------------------------------------------------------------------

def pairwise_llm(result, outcome_label: str) -> list[str]:
    """Bonferroni-corrected pairwise contrasts between LLM levels."""
    import scipy.stats as st

    fe     = result.fe_params
    cov    = result.cov_params()

    # Find llm parameter names
    llm_params = [p for p in fe.index if "C(llm)" in p]
    # levels: reference + one per T.xxx param
    ref_label = REF["llm"]
    other_labels = [p.split("[T.")[-1].rstrip("]") for p in llm_params]
    all_levels = [ref_label] + other_labels

    pairs = []
    for i in range(len(all_levels)):
        for j in range(i + 1, len(all_levels)):
            a, b = all_levels[i], all_levels[j]
            coef_a = fe.get(f"C(llm)[T.{a}]", 0.0)
            coef_b = fe.get(f"C(llm)[T.{b}]", 0.0)
            diff   = coef_b - coef_a

            # Variance of (coef_b - coef_a) from the covariance matrix
            key_a = f"C(llm)[T.{a}]"
            key_b = f"C(llm)[T.{b}]"
            var_a  = cov.loc[key_a, key_a] if key_a in cov.index else 0.0
            var_b  = cov.loc[key_b, key_b] if key_b in cov.index else 0.0
            cov_ab = cov.loc[key_a, key_b] if (key_a in cov.index and key_b in cov.index) else 0.0
            se_diff = np.sqrt(var_a + var_b - 2 * cov_ab)

            z = diff / se_diff if se_diff > 0 else np.nan
            p_raw = 2 * st.norm.sf(abs(z)) if not np.isnan(z) else np.nan
            n_pairs = len(all_levels) * (len(all_levels) - 1) // 2
            p_bonf = min(p_raw * n_pairs, 1.0) if not np.isnan(p_raw) else np.nan

            ci_lo = diff - 1.96 * se_diff
            ci_hi = diff + 1.96 * se_diff
            sig   = "**" if p_bonf < 0.01 else ("*" if p_bonf < 0.05 else "")
            pairs.append((a, b, diff, se_diff, z, p_bonf, ci_lo, ci_hi, sig))

    lines = [f"**Pairwise LLM contrasts ({outcome_label}, Bonferroni corrected)**", ""]
    lines.append("| Comparison | Diff | SE | z | p (adj) | 95% CI | Sig |")
    lines.append("|-----------|------|----|---|---------|--------|-----|")
    for a, b, diff, se, z, p, lo, hi, sig in pairs:
        p_str = "< .001" if p < 0.001 else f"{p:.3f}"
        lines.append(f"| {b} vs {a} | {diff:+.4f} | {se:.4f} | {z:+.2f} | {p_str} | [{lo:+.4f}, {hi:+.4f}] | {sig} |")
    lines.append("")
    return lines


# ---------------------------------------------------------------------------
# Bad question detection
# ---------------------------------------------------------------------------

def bad_questions(result, question_map: dict, threshold_sd: float = BAD_Q_SD_THRESHOLD) -> list[str]:
    """
    Extract per-question BLUPs (random intercepts) from the fitted model.
    Flag questions where BLUP < mean - threshold_sd * std.
    """
    try:
        re = result.random_effects  # dict: {group_id: Series}
    except ValueError:
        return ["### Systematically Hard Questions (BLUPs)", "",
                "⚠ Random effects covariance is singular — question-level variance is negligible "
                "(all questions answered similarly across configs).", ""]
    blups = {int(qnum): float(vals.iloc[0]) for qnum, vals in re.items()}

    values = np.array(list(blups.values()))
    mean_b = np.mean(values)
    std_b  = np.std(values)
    cutoff = mean_b - threshold_sd * std_b

    flagged = sorted([(q, b) for q, b in blups.items() if b < cutoff],
                     key=lambda x: x[1])

    lines = ["### Systematically Hard Questions (BLUPs)", ""]
    lines.append(f"Random intercept stats: mean={mean_b:.3f}, SD={std_b:.3f}, "
                 f"cutoff (<mean−{threshold_sd}×SD) = {cutoff:.3f}")
    lines.append("")

    if not flagged:
        lines.append("No questions flagged as systematically hard.")
    else:
        lines.append("| Q# | BLUP | Question |")
        lines.append("|----|------|---------|")
        for qnum, blup in flagged:
            q_text = question_map.get(qnum, f"Q{qnum}")
            lines.append(f"| {qnum} | {blup:+.3f} | {q_text} |")

    lines.append("")
    return lines


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Mixed effects analysis of RAG sweep CSV")
    parser.add_argument("--input", help="Path to sweep CSV (default: latest in evaluation/results/)")
    args = parser.parse_args()

    df = load_csv(args.input)
    df = add_composites(df)
    question_map = load_questions()

    report_lines = [
        "# Mixed Effects Analysis — RAG Sweep",
        "",
        "Models: `score ~ C(llm) + C(chunk) + C(k) + C(weights) + (1|q_num)`",
        "Random intercept per question removes between-question noise from fixed effect estimates.",
        "Reference levels: llm=qwen3-8b, chunk=600, k=2, weights=pure-sem",
        "",
        "Significance: \\* p<0.05  \\*\\* p<0.01  (pairwise contrasts: Bonferroni corrected)",
        "",
    ]

    # ── Individual metrics ────────────────────────────────────────────────
    report_lines.append("## Individual Metrics")
    report_lines.append("")

    for metric in ALL_METRICS:
        print(f"Fitting model: {metric} ...")
        result = fit_lme(df, metric)
        if result is None:
            continue
        report_lines += fixed_effects_table(result, metric.replace("_", " ").title())
        if metric in GROUP_A:
            report_lines += pairwise_llm(result, metric)

    # ── Group A: LLM Answer Quality ───────────────────────────────────────
    print("Fitting model: composite_A (LLM quality) ...")
    result_A = fit_lme(df, "composite_A")
    report_lines.append("---")
    report_lines.append(f"## Group A — {GROUP_A_LABEL}")
    report_lines.append(f"*Composite of: {', '.join(GROUP_A)} (equal weights, -1 excluded)*")
    report_lines.append("")
    if result_A:
        report_lines += fixed_effects_table(result_A, GROUP_A_LABEL)
        report_lines += pairwise_llm(result_A, "LLM Quality Composite")

    # ── Group B: RAG Retrieval Quality ────────────────────────────────────
    print("Fitting model: composite_B (RAG quality) ...")
    result_B = fit_lme(df, "composite_B")
    report_lines.append("---")
    report_lines.append(f"## Group B — {GROUP_B_LABEL}")
    report_lines.append(f"*Composite of: {', '.join(GROUP_B)} (equal weights, -1 excluded)*")
    report_lines.append("")
    if result_B:
        report_lines += fixed_effects_table(result_B, GROUP_B_LABEL)

    # ── Overall weighted score ─────────────────────────────────────────────
    print("Fitting model: composite_overall ...")
    result_overall = fit_lme(df, "composite_overall")
    report_lines.append("---")
    report_lines.append("## Overall Weighted Score")
    w_str = " + ".join(f"{v}×{k}" for k, v in WEIGHTS.items())
    report_lines.append(f"*{w_str} (renormalised if metric unavailable)*")
    report_lines.append("")
    if result_overall:
        report_lines += fixed_effects_table(result_overall, "Overall Weighted Score")
        report_lines += pairwise_llm(result_overall, "Overall")

        # ── Bad question detection ─────────────────────────────────────────
        report_lines.append("---")
        report_lines.append("## Bad Question Detection")
        report_lines.append("")
        report_lines += bad_questions(result_overall, question_map)

    # ── Save report ───────────────────────────────────────────────────────
    Path(RESULTS_DIR).mkdir(parents=True, exist_ok=True)
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines) + "\n")

    print(f"\nReport saved → {REPORT_PATH}")


if __name__ == "__main__":
    main()
