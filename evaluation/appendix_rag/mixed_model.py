# evaluation/appendix_rag/mixed_model.py
"""
Linear Mixed Effects analysis of Retrieved vs Random Chunks ablation results.

One LME per metric (all five) with random intercept per question. The fixed-effect
coefficient on `condition=rag` is the estimated mean difference on the 0-1 scale —
directly interpretable as retrieval quality gain.

Model:  score ~ C(condition) + (1|q_num)
Ref:    condition=random_chunks

Family-wise correction: Holm across all five metrics.

Key metric for interpretation:
  contextual_recall — primary retrieval quality signal. Random chunks → near zero
    because the ground-truth information is not in the context. Retrieved chunks →
    high because the retriever found relevant material. The gap between conditions
    directly quantifies retrieval value.
  faithfulness — note inverted semantics for random_chunks: high faithfulness there
    means the model confabulated answers consistent with irrelevant context (bad).
    Low faithfulness for random_chunks means the model ignored noise and used its
    own knowledge. Interpret direction accordingly.

Outputs:
  A. Descriptive means per condition (all five metrics).
  B. Fixed effects table + Wald test per metric.
  C. Holm-adjusted p-values across all five metrics.
  D. Bad-question detection via BLUPs (1.5 SD threshold) anchored on
     correctness LME.
  E. Robustness re-run: LME refit after dropping flagged questions.

Usage:
    python evaluation/appendix_rag/mixed_model.py
    python evaluation/appendix_rag/mixed_model.py --input evaluation/results/rag_ablation_results.json
"""

import os
import sys
import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
_EVAL = os.path.dirname(_HERE)
_ROOT = os.path.dirname(_EVAL)
sys.path.insert(0, _EVAL)
sys.path.insert(0, os.path.join(_EVAL, "shared"))

RESULTS_DIR   = os.path.join(_EVAL, "results")
REPORT_PATH   = os.path.join(RESULTS_DIR, "reports", "rag_ablation_mixed_model.md")
REF_CONDITION = "random_chunks"
BAD_Q_SD_THRESHOLD = 1.5

ALL_METRICS = [
    "answer_relevancy",
    "actionability",
    "correctness",
    "faithfulness",
    "contextual_recall",
]

FAITHFULNESS_NOTE = (
    "**Faithfulness interpretation note:** For `random_chunks`, high faithfulness "
    "indicates the model produced claims consistent with irrelevant context "
    "(confabulation from noise). Low faithfulness means the model ignored "
    "irrelevant context and drew on its own knowledge. For `rag`, high faithfulness "
    "is the desired outcome — the model stays grounded in relevant retrieved material."
)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_data(input_path: str = None) -> pd.DataFrame:
    if input_path is None:
        input_path = os.path.join(RESULTS_DIR, "rag_ablation_results.json")
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Not found: {input_path}")
    print(f"Loading: {input_path}\n")

    import json
    with open(input_path, encoding="utf-8") as f:
        raw = json.load(f)

    conditions = raw.get("conditions", raw) if isinstance(raw, dict) else raw

    rows = []
    for condition, results in conditions.items():
        for q_num, r in enumerate(results, start=1):
            row = {
                "condition": condition,
                "q_num":     q_num,
                "question":  r.get("question", ""),
                "category":  r.get("category", ""),
            }
            for m in ALL_METRICS:
                v = r.get(m)
                row[m] = float(v) if v is not None else np.nan
            rows.append(row)
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


def fit_lme(df: pd.DataFrame, outcome: str, exclude_q: list = None):
    try:
        import statsmodels.formula.api as smf
    except ImportError:
        raise ImportError("statsmodels required: pip install statsmodels")

    sub = df[["q_num", "condition", outcome]].dropna().copy()
    sub = sub[sub[outcome] >= 0]
    if exclude_q:
        sub = sub[~sub["q_num"].isin(exclude_q)]

    if len(sub) < 10:
        print(f"  ⚠  Too few rows for {outcome} ({len(sub)}) — skipping")
        return None

    sub["condition"] = pd.Categorical(
        sub["condition"],
        categories=_ordered_cats(sub["condition"], REF_CONDITION),
    )

    formula = f"{outcome} ~ C(condition)"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = smf.mixedlm(formula, data=sub, groups=sub["q_num"])
        try:
            result = model.fit(reml=True, method="powell")
        except Exception:
            result = model.fit(reml=True, method="lbfgs")

    sigma_q = result.cov_re.values[0][0]
    print(f"  {outcome}: var_q = {sigma_q:.6f}  (n={int(result.nobs)})")
    return result


# ---------------------------------------------------------------------------
# Fixed effects + Wald
# ---------------------------------------------------------------------------

def _clean_param(s: str) -> str:
    return s.replace("C(condition)[T.", "condition=").rstrip("]")


def fixed_effects_table(result, title: str) -> list:
    fe    = result.fe_params
    ci    = result.conf_int()
    pvals = result.pvalues
    sigma_q = result.cov_re.values[0][0]

    lines = [f"### {title}", ""]
    lines.append(f"Random effects variance (σ²_q): {sigma_q:.6f}")
    lines.append(f"N observations: {int(result.nobs)}")
    lines.append("")
    lines.append("| Parameter | Coef (Δ) | SE | z | p (Wald) | 95% CI |")
    lines.append("|-----------|----------|----|---|----------|--------|")

    for param in fe.index:
        if param == "Intercept":
            intercept_p = "< .001" if pvals[param] < 0.001 else f"{pvals[param]:.3f}"
            lines.append(
                f"| Intercept ({REF_CONDITION}) | {fe[param]:.4f} "
                f"| {result.bse[param]:.4f} | {result.tvalues[param]:+.2f} "
                f"| {intercept_p} | [{ci.loc[param, 0]:+.4f}, {ci.loc[param, 1]:+.4f}] |"
            )
            continue
        coef = fe[param]
        se   = result.bse[param]
        z    = result.tvalues[param]
        p    = pvals[param]
        lo   = ci.loc[param, 0]
        hi   = ci.loc[param, 1]
        p_str = "< .001" if p < 0.001 else f"{p:.3f}"
        label = _clean_param(param)
        lines.append(
            f"| {label} | {coef:+.4f} | {se:.4f} | {z:+.2f} | {p_str} "
            f"| [{lo:+.4f}, {hi:+.4f}] |"
        )
    lines.append("")
    return lines


def condition_p_value(result) -> tuple:
    for param in result.fe_params.index:
        if param == "Intercept":
            continue
        return float(result.fe_params[param]), float(result.pvalues[param])
    return (np.nan, np.nan)


# ---------------------------------------------------------------------------
# Holm correction
# ---------------------------------------------------------------------------

def holm_correct(pvals: dict) -> dict:
    items = [(m, p) for m, p in pvals.items() if not np.isnan(p)]
    items.sort(key=lambda x: x[1])
    m = len(items)
    adj = {}
    running_max = 0.0
    for i, (metric, p) in enumerate(items):
        factor = m - i
        p_adj = min(max(p * factor, running_max), 1.0)
        running_max = p_adj
        adj[metric] = p_adj
    return adj


# ---------------------------------------------------------------------------
# Bad-question BLUPs
# ---------------------------------------------------------------------------

def bad_questions(result, question_map: dict) -> tuple:
    try:
        re = result.random_effects
    except ValueError:
        return [], [
            "### Systematically Hard Questions (BLUPs)", "",
            "⚠ Random effects covariance singular — negligible question-level variance.", "",
        ]

    blups = {int(qnum): float(vals.iloc[0]) for qnum, vals in re.items()}
    values = np.array(list(blups.values()))
    mean_b = float(np.mean(values))
    std_b  = float(np.std(values))
    cutoff = mean_b - BAD_Q_SD_THRESHOLD * std_b

    flagged = sorted([(q, b) for q, b in blups.items() if b < cutoff], key=lambda x: x[1])

    lines = ["### Systematically Hard Questions (BLUPs)", ""]
    lines.append(
        f"Random intercept stats (anchored on correctness LME): "
        f"mean={mean_b:.3f}, SD={std_b:.3f}, "
        f"cutoff (< mean − {BAD_Q_SD_THRESHOLD}×SD) = {cutoff:.3f}"
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
    return [q for q, _ in flagged], lines


# ---------------------------------------------------------------------------
# Descriptive summary
# ---------------------------------------------------------------------------

def _valid(series: pd.Series) -> pd.Series:
    return series[(series.notna()) & (series >= 0)]


def descriptive_table(df: pd.DataFrame) -> list:
    lines = [
        "## Descriptive Summary", "",
        "All five metrics apply to both conditions (both have retrieval_context).", "",
    ]
    header = "| Condition | N | " + " | ".join(m.replace("_", " ").title() for m in ALL_METRICS) + " |"
    sep    = "|-----------|---|" + "|".join("-----" for _ in ALL_METRICS) + "|"
    lines.append(header)
    lines.append(sep)

    conds = [REF_CONDITION] + sorted(set(df["condition"].unique()) - {REF_CONDITION})
    for condition in conds:
        sub = df[df["condition"] == condition]
        n   = len(sub)
        cells = [condition, str(n)]
        for m in ALL_METRICS:
            vals = _valid(sub[m])
            cells.append(f"{vals.mean():.4f}" if len(vals) > 0 else "—")
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")
    lines.append(FAITHFULNESS_NOTE)
    lines.append("")
    return lines


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Mixed effects analysis of retrieval ablation results")
    parser.add_argument("--input", help="Path to rag_ablation_results.json")
    args = parser.parse_args()

    df = load_data(args.input)
    question_map = load_questions()

    report = [
        "# Mixed Effects Analysis — Retrieved vs Random Chunks Ablation",
        "",
        "**Research question:** Does targeted retrieval (hybrid 70/30 BM25+semantic) find "
        "meaningfully better context than chance, measured by downstream answer quality?",
        "",
        "Model: `score ~ C(condition) + (1|q_num)` (random intercept per question removes "
        "between-question difficulty noise from the condition effect.)",
        f"Reference level: condition=`{REF_CONDITION}`",
        "The coefficient `condition=rag` is the estimated mean gain on the 0–1 metric scale.",
        "Family-wise correction: Holm across all five metrics.",
        "",
        "**Key metric:** `contextual_recall` — directly measures whether the retrieved context "
        "contains the information needed to answer each question. Random chunks → near zero "
        "(corpus is large, relevant chunks rare by chance). Retrieved chunks → high (retriever "
        "selects relevant material). The gap is the core empirical claim.",
        "",
    ]

    report += descriptive_table(df)

    report.append("## LME Results — All Metrics")
    report.append("")

    raw_p = {}
    coefs = {}
    combined_result = None

    for metric in ALL_METRICS:
        print(f"Fitting LME: {metric} ...")
        result = fit_lme(df, metric)
        if result is None:
            continue
        title = metric.replace("_", " ").title()
        report += fixed_effects_table(result, title)
        coef, p = condition_p_value(result)
        raw_p[metric] = p
        coefs[metric] = coef
        if metric == "correctness":
            combined_result = result

    adj_p = holm_correct(raw_p)

    report.append("### Holm-corrected family-wise p-values")
    report.append("")
    report.append("| Metric | Δ (rag − random_chunks) | Wald p (raw) | Holm-adjusted p |")
    report.append("|--------|-------------------------|--------------|-----------------|")
    for m in ALL_METRICS:
        if m not in raw_p:
            continue
        rp = raw_p[m]; ap = adj_p[m]; c = coefs[m]
        rp_str = "< .001" if rp < 0.001 else f"{rp:.3f}"
        ap_str = "< .001" if ap < 0.001 else f"{ap:.3f}"
        report.append(f"| {m} | {c:+.4f} | {rp_str} | {ap_str} |")
    report.append("")
    report.append(FAITHFULNESS_NOTE)
    report.append("")

    report.append("---")
    report.append("## Bad Question Detection")
    report.append("")
    flagged_q, blup_lines = bad_questions(combined_result, question_map) if combined_result else ([], [])
    report += blup_lines

    if flagged_q:
        report.append("## Robustness Re-run (flagged questions dropped)")
        report.append("")
        report.append(
            f"Dropping Q#s {sorted(flagged_q)} and refitting all metrics. "
            "Checks whether the main effect is driven by systematically hard questions."
        )
        report.append("")
        rob_p = {}
        rob_coefs = {}
        for metric in ALL_METRICS:
            print(f"Refitting LME (robustness): {metric} ...")
            res = fit_lme(df, metric, exclude_q=flagged_q)
            if res is None:
                continue
            c, p = condition_p_value(res)
            rob_p[metric] = p
            rob_coefs[metric] = c
        rob_adj = holm_correct(rob_p)

        report.append("| Metric | Δ (rag − random_chunks) | Wald p (raw) | Holm-adjusted p |")
        report.append("|--------|-------------------------|--------------|-----------------|")
        for m in ALL_METRICS:
            if m not in rob_p:
                continue
            rp = rob_p[m]; ap = rob_adj[m]; c = rob_coefs[m]
            rp_str = "< .001" if rp < 0.001 else f"{rp:.3f}"
            ap_str = "< .001" if ap < 0.001 else f"{ap:.3f}"
            report.append(f"| {m} | {c:+.4f} | {rp_str} | {ap_str} |")
        report.append("")

    os.makedirs(os.path.join(RESULTS_DIR, "reports"), exist_ok=True)
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(report) + "\n")

    print(f"\nReport saved -> {REPORT_PATH}")


if __name__ == "__main__":
    main()
