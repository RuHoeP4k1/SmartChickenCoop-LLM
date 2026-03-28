# evaluation/prompt_variant_mixed_model.py
"""
Linear Mixed Effects analysis of prompt variant evaluation results.

Fits one LME model per metric with a random intercept per question (q_num)
to remove between-question noise from prompt effect estimates.

Model: score ~ C(variant) + (1|q_num)
Reference level: baseline

Outputs:
  A. Fixed effects per metric (actionability, correctness, combined)
  B. Pairwise variant contrasts (Bonferroni-corrected)
  C. Bad question detection via BLUPs

Usage:
    python evaluation/prompt_variant_mixed_model.py
    python evaluation/prompt_variant_mixed_model.py --input evaluation/results/prompt_variant_results.json
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
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _HERE)

RESULTS_DIR  = os.path.join(_HERE, "results")
REPORT_PATH  = os.path.join(RESULTS_DIR, "prompt_variant_mixed_model.md")
REF_VARIANT  = "baseline"
BAD_Q_SD_THRESHOLD = 1.5
METRICS = ["actionability", "correctness", "combined"]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_data(input_path: str = None) -> pd.DataFrame:
    if input_path is None:
        input_path = os.path.join(RESULTS_DIR, "prompt_variant_results.json")
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Not found: {input_path}")
    print(f"Loading: {input_path}\n")

    import json
    with open(input_path) as f:
        raw = json.load(f)

    rows = []
    for variant, results in raw.items():
        for q_num, r in enumerate(results, start=1):
            action = float(r.get("actionability", -1))
            correct = float(r.get("correctness", -1))
            combined = np.nan
            if action >= 0 and correct >= 0:
                combined = (action + correct) / 2
            elif action >= 0:
                combined = action
            elif correct >= 0:
                combined = correct
            rows.append({
                "variant": variant,
                "q_num":   q_num,
                "question": r.get("question", ""),
                "category": r.get("category", ""),
                "actionability": action if action >= 0 else np.nan,
                "correctness":   correct if correct >= 0 else np.nan,
                "combined":      combined,
            })

    return pd.DataFrame(rows)


def load_questions() -> dict:
    try:
        from evaluation_data import TEST_CASES
        return {i + 1: tc["question"] for i, tc in enumerate(TEST_CASES)}
    except ImportError:
        return {}


# ---------------------------------------------------------------------------
# Mixed model fitting
# ---------------------------------------------------------------------------

def _ordered_cats(series, ref):
    others = sorted(set(series.astype(str).unique()) - {str(ref)})
    return [str(ref)] + others


def fit_lme(df: pd.DataFrame, outcome: str):
    try:
        import statsmodels.formula.api as smf
    except ImportError:
        raise ImportError("statsmodels required: pip install statsmodels")

    sub = df[["q_num", "variant", outcome]].dropna().copy()
    sub = sub[sub[outcome] >= 0]

    if len(sub) < 10:
        print(f"  ⚠  Too few rows for {outcome} ({len(sub)}) — skipping")
        return None

    sub["variant"] = pd.Categorical(
        sub["variant"],
        categories=_ordered_cats(sub["variant"], REF_VARIANT)
    )

    formula = f"{outcome} ~ C(variant)"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = smf.mixedlm(formula, data=sub, groups=sub["q_num"])
        try:
            result = model.fit(reml=True, method="powell")
        except Exception:
            result = model.fit(reml=True, method="lbfgs")

    sigma_q = result.cov_re.values[0][0]
    print(f"  σ²_q = {sigma_q:.6f}  (n={int(result.nobs)})")
    return result


# ---------------------------------------------------------------------------
# Fixed effects table
# ---------------------------------------------------------------------------

def _clean_param(s: str) -> str:
    return s.replace("C(variant)[T.", "variant=").rstrip("]")


def fixed_effects_table(result, title: str) -> list:
    fe   = result.fe_params
    ci   = result.conf_int()
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
        lines.append(
            f"| {label} | {coef:+.4f} | {se:.4f} | {z:+.2f} | {p_str} "
            f"| [{lo:+.4f}, {hi:+.4f}] | {sig} |"
        )
    lines.append("")
    return lines


# ---------------------------------------------------------------------------
# Pairwise variant contrasts
# ---------------------------------------------------------------------------

def pairwise_variants(result, outcome_label: str) -> list:
    import scipy.stats as st

    fe  = result.fe_params
    cov = result.cov_params()

    variant_params = [p for p in fe.index if "C(variant)" in p]
    ref_label      = REF_VARIANT
    other_labels   = [p.split("[T.")[-1].rstrip("]") for p in variant_params]
    all_levels     = [ref_label] + other_labels

    pairs_data = []
    for i in range(len(all_levels)):
        for j in range(i + 1, len(all_levels)):
            a, b   = all_levels[i], all_levels[j]
            key_a  = f"C(variant)[T.{a}]"
            key_b  = f"C(variant)[T.{b}]"
            coef_a = fe.get(key_a, 0.0)
            coef_b = fe.get(key_b, 0.0)
            diff   = coef_b - coef_a

            var_a  = cov.loc[key_a, key_a] if key_a in cov.index else 0.0
            var_b  = cov.loc[key_b, key_b] if key_b in cov.index else 0.0
            cov_ab = cov.loc[key_a, key_b] if (key_a in cov.index and key_b in cov.index) else 0.0
            se_diff = np.sqrt(max(var_a + var_b - 2 * cov_ab, 0))

            z     = diff / se_diff if se_diff > 0 else np.nan
            p_raw = 2 * st.norm.sf(abs(z)) if not np.isnan(z) else np.nan
            n_p   = len(all_levels) * (len(all_levels) - 1) // 2
            p_bonf = min(p_raw * n_p, 1.0) if not np.isnan(p_raw) else np.nan

            ci_lo = diff - 1.96 * se_diff
            ci_hi = diff + 1.96 * se_diff
            sig   = "**" if p_bonf < 0.01 else ("*" if p_bonf < 0.05 else "")
            pairs_data.append((a, b, diff, se_diff, z, p_bonf, ci_lo, ci_hi, sig))

    lines = [f"**Pairwise variant contrasts ({outcome_label}, Bonferroni corrected)**", ""]
    lines.append("| Comparison | Diff | SE | z | p (adj) | 95% CI | Sig |")
    lines.append("|-----------|------|----|---|---------|--------|-----|")
    for a, b, diff, se, z, p, lo, hi, sig in pairs_data:
        p_str = "< .001" if p < 0.001 else f"{p:.3f}"
        lines.append(
            f"| {b} vs {a} | {diff:+.4f} | {se:.4f} | {z:+.2f} | {p_str} "
            f"| [{lo:+.4f}, {hi:+.4f}] | {sig} |"
        )
    lines.append("")
    return lines


# ---------------------------------------------------------------------------
# Bad question detection
# ---------------------------------------------------------------------------

def bad_questions(result, question_map: dict) -> list:
    try:
        re = result.random_effects
    except ValueError:
        return ["### Systematically Hard Questions (BLUPs)", "",
                "⚠ Random effects covariance singular — negligible question-level variance.", ""]

    blups  = {int(qnum): float(vals.iloc[0]) for qnum, vals in re.items()}
    values = np.array(list(blups.values()))
    mean_b = np.mean(values)
    std_b  = np.std(values)
    cutoff = mean_b - BAD_Q_SD_THRESHOLD * std_b

    flagged = sorted([(q, b) for q, b in blups.items() if b < cutoff], key=lambda x: x[1])

    lines = ["### Systematically Hard Questions (BLUPs)", ""]
    lines.append(
        f"Random intercept stats: mean={mean_b:.3f}, SD={std_b:.3f}, "
        f"cutoff (<mean−{BAD_Q_SD_THRESHOLD}×SD) = {cutoff:.3f}"
    )
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
# Descriptive summary
# ---------------------------------------------------------------------------

def descriptive_table(df: pd.DataFrame) -> list:
    lines = ["## Descriptive Summary", ""]
    lines.append("| Variant | N | Actionability (mean) | Correctness (mean) | Combined (mean) |")
    lines.append("|---------|---|----------------------|--------------------|-----------------|")
    for variant in sorted(df["variant"].unique()):
        sub = df[df["variant"] == variant]
        n   = len(sub)
        a   = sub["actionability"].dropna().mean()
        c   = sub["correctness"].dropna().mean()
        comb = sub["combined"].dropna().mean()
        lines.append(f"| {variant} | {n} | {a:.4f} | {c:.4f} | {comb:.4f} |")
    lines.append("")
    return lines


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Mixed effects analysis of prompt variant results")
    parser.add_argument("--input", help="Path to prompt_variant_results.json")
    args = parser.parse_args()

    df = load_data(args.input)
    question_map = load_questions()

    report_lines = [
        "# Mixed Effects Analysis — Prompt Variants",
        "",
        "Model: `score ~ C(variant) + (1|q_num)`",
        "Random intercept per question removes between-question noise from variant effect estimates.",
        f"Reference level: variant={REF_VARIANT}",
        "",
        "Significance: \\* p<0.05  \\*\\* p<0.01  (pairwise contrasts: Bonferroni corrected)",
        "",
    ]

    report_lines += descriptive_table(df)

    report_lines.append("## Individual Metrics")
    report_lines.append("")

    result_combined = None
    for metric in METRICS:
        print(f"Fitting model: {metric} ...")
        result = fit_lme(df, metric)
        if result is None:
            continue
        title = metric.replace("_", " ").title()
        report_lines += fixed_effects_table(result, title)
        report_lines += pairwise_variants(result, title)
        if metric == "combined":
            result_combined = result

    if result_combined:
        report_lines.append("---")
        report_lines.append("## Bad Question Detection")
        report_lines.append("")
        report_lines += bad_questions(result_combined, question_map)

    Path(RESULTS_DIR).mkdir(parents=True, exist_ok=True)
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines) + "\n")

    print(f"\nReport saved → {REPORT_PATH}")


if __name__ == "__main__":
    main()
