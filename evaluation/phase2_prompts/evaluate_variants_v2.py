"""
evaluate_prompt_variants_v2.py — Phase 2b validation

Compares the 3 original prompt variants (structured, concise, expert) against
the current production frankenstein prompt using the 45 corpus-grounded
synthetic goldens. This is a validation experiment — it asks "does the
frankenstein actually beat the variants it was composed from?"

Design:
  Conditions:  structured | concise | expert | frankenstein (production)
  Questions:   45 synthetic goldens from goldens_synth_v1.json (RAG-grounded)
  RAG:         on (hybrid 70/30, k=4, chunk=1000 — Phase 1 winner config)
  Sensors:     off (use_sensors=False — isolates prompt effect)
  Metrics:     Actionability (G-Eval), Correctness (G-Eval)
  Judge:       gpt-4o-mini via OpenRouter (SWEEP_JUDGE_MODEL)
  Retries:     5 attempts per metric with 1.5x linear backoff
  Reference:   frankenstein (for LME contrast)

Output:
  results/prompt_variant_v2_results.json  — raw per-question scores
  results/prompt_variant_v2_mixed_model.md — LME summary

Usage:
    python evaluation/evaluate_prompt_variants_v2.py
    python evaluation/evaluate_prompt_variants_v2.py --n 5    # smoke test
"""

import os
import sys
import json
import time
import argparse
import warnings
from pathlib import Path
from typing import Dict, List

os.environ["DEEPEVAL_TELEMETRY_OPT_OUT"] = "YES"

_HERE = os.path.dirname(os.path.abspath(__file__))
_EVAL = os.path.dirname(_HERE)
_ROOT = os.path.dirname(_EVAL)
sys.path.insert(0, _ROOT)
sys.path.insert(0, _EVAL)

from dotenv import load_dotenv
load_dotenv(dotenv_path=os.path.join(_ROOT, ".env"))

from langchain_openai import ChatOpenAI
from deepeval.models.base_model import DeepEvalBaseLLM
from deepeval.metrics.g_eval.utils import Rubric
from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCase, LLMTestCaseParams

from backend.rag_functions import (
    load_documents, split_documents, build_vector_store,
    build_bm25_retriever, hybrid_search,
    format_context, generate_response,
)
from backend.prompts import _SAFETY_RULES, STANDARD_PROMPT
from eval_config import ACTIONABILITY_CRITERIA, CORRECTNESS_CRITERIA

GOLDENS_PATH = os.path.join(_EVAL, "data", "goldens_synth_v1.json")
RESULTS_DIR  = os.path.join(_EVAL, "results")
OUT_JSON     = os.path.join(RESULTS_DIR, "prompt_variant_v2_results.json")
OUT_MD       = os.path.join(RESULTS_DIR, "reports", "prompt_variant_v2_mixed_model.md")

JUDGE_MODEL  = os.getenv("SWEEP_JUDGE_MODEL", "openai/gpt-4o-mini")
REF_VARIANT  = "frankenstein"

# Frankenstein = STANDARD_PROMPT with sensor slots blanked out
_FRANKENSTEIN_TEMPLATE = STANDARD_PROMPT.replace(
    "{safety_rules}", _SAFETY_RULES
).replace(
    "{sensor_section}", ""
).replace(
    "{sensor_anchor}", ""
).replace(
    "{sensor_prioritise}", ""
)

PROMPT_VARIANTS: Dict[str, str] = {
    "structured": """You are a practical chicken-keeping advisor — the kind of knowledgeable friend every flock owner wishes they had. You cut through the noise and give keepers exactly what they need to act on, organized so nothing gets missed.

Your rules (non-negotiable):
- Never suggest medications, dosages, or chemical treatments.
- Metric units only: °C, kg, cm, litres. Convert any imperial values from the source material before answering.
- Use plain language — your reader is a hobby keeper, not a vet student.

--- Reference knowledge (background information only) ---
{context}
--- End of reference knowledge ---

Question: {query}

Give your answer in this structure — it makes the advice easier to act on:

**The short answer:** One or two sentences. Direct. No hedging.

**Steps to take:**
1. First action (specific — what exactly, not just "check the coop")
2. Second action
3. What to watch for and over what timeframe
(If the question is purely factual with nothing to act on, write "No action needed — this is background knowledge.")

**When to call a vet or experienced keeper:** Name the specific warning signs. Only include this section if the question touches health or injury — do not add it as a default safety disclaimer.""",

    "concise": """You are a knowledgeable, practical assistant for hobby chicken keepers. Give direct, plain-language advice. Never suggest medications or chemical treatments. Use metric units (°C, kg, cm, litres).

--- Reference knowledge (background information only) ---
{context}
--- End of reference knowledge ---

Question: {query}

Answer concisely based on the knowledge above. Mention a vet only if the question involves illness or injury.""",

    "expert": """You are an experienced poultry specialist — part researcher, part seasoned flock keeper. You have spent years studying poultry health and husbandry, and just as many years applying it with real backyard flocks. You know the difference between textbook advice and what actually works at 6 a.m. on a cold morning.

You give advice the way an expert gives advice to a friend: direct, accurate, grounded in evidence, and calibrated to what a hobby keeper can realistically do. You do not inflate risks, add unnecessary caveats, or recommend vets for problems an experienced keeper handles routinely.

Constraints:
- Never suggest medications, dosages, or chemical treatments.
- Use metric units (°C, kg, cm, litres). Convert any imperial figures from source material before answering.

--- Reference knowledge (background information only) ---
{context}
--- End of reference knowledge ---

Question: {query}

Answer based on the knowledge above. Be direct and complete — give the keeper enough to act confidently. Where the knowledge base leaves a gap, draw on sound general principles rather than refusing to answer. If the question involves genuine illness or injury, specify which signs would warrant professional advice.""",

    "frankenstein": _FRANKENSTEIN_TEMPLATE,
}


# =============================================================================
# Judge
# =============================================================================

class OpenRouterJudge(DeepEvalBaseLLM):
    def __init__(self, model_name: str = JUDGE_MODEL):
        self.model_name = model_name
        self.model = ChatOpenAI(
            model=model_name,
            base_url="https://openrouter.ai/api/v1",
            api_key=os.getenv("OPENROUTER_API_KEY"),
            temperature=0,
            max_tokens=2048,
        )

    def load_model(self):
        return self.model

    def generate(self, prompt: str) -> str:
        response = self.model.invoke(prompt)
        return response.content if hasattr(response, "content") else str(response)

    async def a_generate(self, prompt: str) -> str:
        return self.generate(prompt)

    def get_model_name(self) -> str:
        return f"OpenRouter/{self.model_name}"


# =============================================================================
# Rubrics (identical to evaluate_prompt_variants.py)
# =============================================================================

_ACTION_RUBRIC = [
    Rubric(score_range=(0, 3), expected_outcome="Not actionable — vague generalities, nothing to act on, repeats the question, or just restates the problem."),
    Rubric(score_range=(4, 5), expected_outcome="Partially actionable — contains some useful information but missing key specifics, overly hedged, or leaves important gaps the user must fill themselves."),
    Rubric(score_range=(6, 7), expected_outcome="Helpful — answers the question clearly with at least one concrete action or specific fact; a hobby keeper can act on it, though minor gaps or hedging remain."),
    Rubric(score_range=(8, 10), expected_outcome="Precisely actionable — directly targeted to the specific question, covers all key steps or facts with appropriate detail, calibrated to the user's actual situation, no significant gaps or unnecessary hedging. Reserve 9-10 for answers that are both complete AND notably well-tailored."),
]

_CORRECT_RUBRIC = [
    Rubric(score_range=(0, 3), expected_outcome="Incorrect or harmful — contains wrong facts that could mislead a beginner, dangerous advice, or actively contradicts the reference answer or established poultry practice."),
    Rubric(score_range=(4, 5), expected_outcome="Partially correct — the core claim is broadly right but misses important details present in the reference answer, contains a meaningful inaccuracy, or is so hedged it becomes unreliable."),
    Rubric(score_range=(6, 7), expected_outcome="Correct — accurate and consistent with standard practice and the reference answer on main points; minor omissions or imprecisions acceptable, but nothing that would mislead the user."),
    Rubric(score_range=(8, 10), expected_outcome="Fully correct — accurate, complete, aligns closely with the reference answer on all key points. Calibrated detail, no significant omissions or inaccuracies. Reserve 9-10 for answers that match both the facts AND the practical emphasis of the reference."),
]


# =============================================================================
# Retry wrapper
# =============================================================================

def _measure_with_retry(metric, test_case, max_attempts: int = 5) -> float:
    delay = 2.0
    for attempt in range(max_attempts):
        try:
            metric.measure(test_case)
            score = metric.score
            if score is not None and 0.0 <= score <= 1.0:
                return float(score)
        except Exception as exc:
            if attempt == max_attempts - 1:
                print(f"    [WARN] {metric.__class__.__name__} failed after {max_attempts} attempts: {exc}")
                return -1.0
        time.sleep(delay)
        delay *= 1.5
    return -1.0


# =============================================================================
# Golden loader
# =============================================================================

def load_goldens(path: str = GOLDENS_PATH) -> List[Dict]:
    with open(path, encoding="utf-8") as f:
        payload = json.load(f)
    rows = payload["goldens"] if "goldens" in payload else payload
    return [
        {
            "question":    r["question"],
            "ground_truth": r.get("ground_truth", ""),
            "category":    r.get("category", "synthetic"),
        }
        for r in rows
    ]


# =============================================================================
# Run one variant
# =============================================================================

def run_variant(
    variant_name: str,
    prompt_template: str,
    questions: List[Dict],
    vectordb,
    bm25_retriever,
    actionability_metric,
    correctness_metric,
    k: int = 4,
    weights: list = None,
    llm_model: str = None,
) -> List[Dict]:
    if weights is None:
        weights = [0.7, 0.3]

    results = []
    for i, test in enumerate(questions, 1):
        question    = test["question"]
        ground_truth = test.get("ground_truth", "")
        category    = test.get("category", "synthetic")

        docs    = hybrid_search(vectordb, bm25_retriever, question, k=k, weights=weights)
        context = format_context(docs)
        prompt  = prompt_template.format(context=context, query=question)

        t0 = time.time()
        answer = generate_response(prompt, model=llm_model)
        elapsed = time.time() - t0

        tc = LLMTestCase(
            input=question,
            actual_output=answer,
            expected_output=ground_truth,
        )

        action_score  = _measure_with_retry(actionability_metric, tc)
        correct_score = _measure_with_retry(correctness_metric, tc)

        results.append({
            "variant":      variant_name,
            "q_num":        i,
            "question":     question,
            "category":     category,
            "ground_truth": ground_truth,
            "answer":       answer,
            "time":         round(elapsed, 2),
            "actionability": action_score,
            "correctness":  correct_score,
        })

        print(f"  [{i:02d}/{len(questions)}] action={action_score:.2f}  "
              f"correct={correct_score:.2f}  {elapsed:.1f}s")

    return results


# =============================================================================
# LME analysis (inline — frankenstein as reference)
# =============================================================================

def _run_lme(all_results: Dict[str, List[Dict]]) -> str:
    try:
        import numpy as np
        import pandas as pd
        import statsmodels.formula.api as smf
    except ImportError:
        return "statsmodels / pandas not installed — skipping LME.\n"

    rows = []
    for variant, results in all_results.items():
        for r in results:
            rows.append({
                "variant":      variant,
                "q_num":        r["q_num"],
                "actionability": float(r["actionability"]) if r["actionability"] >= 0 else np.nan,
                "correctness":  float(r["correctness"])  if r["correctness"]  >= 0 else np.nan,
            })
    df = pd.DataFrame(rows)

    variants_ordered = [REF_VARIANT] + sorted(set(df["variant"].unique()) - {REF_VARIANT})
    df["variant"] = pd.Categorical(df["variant"], categories=variants_ordered)

    metrics = ["actionability", "correctness"]
    raw_p   = {}
    coefs   = {}
    fe_blocks = []

    for metric in metrics:
        sub = df[["q_num", "variant", metric]].dropna()
        sub = sub[sub[metric] >= 0]
        if len(sub) < 10:
            continue
        formula = f"{metric} ~ C(variant)"
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mdl = smf.mixedlm(formula, data=sub, groups=sub["q_num"])
            try:
                res = mdl.fit(reml=True, method="powell")
            except Exception:
                res = mdl.fit(reml=True, method="lbfgs")

        fe    = res.fe_params
        ci    = res.conf_int()
        pvals = res.pvalues
        bse   = res.bse

        lines = [f"### {metric.replace('_', ' ').title()}", ""]
        lines.append(f"| Parameter | Coef (delta) | SE | z | p (Wald) | 95% CI |")
        lines.append("|-----------|-------------|----|----|----------|--------|")
        for param in fe.index:
            label = param if param == "Intercept" else (
                param.replace("C(variant)[T.", "variant=").rstrip("]")
            )
            p_str = "< .001" if pvals[param] < 0.001 else f"{pvals[param]:.3f}"
            lines.append(
                f"| {label} | {fe[param]:+.4f} | {bse[param]:.4f} "
                f"| {res.tvalues[param]:+.2f} | {p_str} "
                f"| [{ci.loc[param, 0]:+.4f}, {ci.loc[param, 1]:+.4f}] |"
            )
        lines.append("")
        fe_blocks.append("\n".join(lines))

        for param in fe.index:
            if param != "Intercept":
                # collect p-values for all non-ref contrasts
                label = param.replace("C(variant)[T.", "").rstrip("]")
                raw_p[f"{metric}:{label}"] = float(pvals[param])
                coefs[f"{metric}:{label}"] = float(fe[param])

    # Holm correction across all non-intercept contrasts
    items = sorted(raw_p.items(), key=lambda x: x[1])
    m = len(items)
    adj_p = {}
    running_max = 0.0
    for i, (key, p) in enumerate(items):
        p_adj = min(max(p * (m - i), running_max), 1.0)
        running_max = p_adj
        adj_p[key] = p_adj

    # Descriptive table
    desc_lines = ["## Descriptive Means", ""]
    desc_lines.append("| Variant | N | Actionability | Correctness |")
    desc_lines.append("|---------|---|---------------|-------------|")
    for v in variants_ordered:
        sub = df[df["variant"] == v]
        n = len(sub)
        act = sub["actionability"].dropna()
        cor = sub["correctness"].dropna()
        a_str = f"{act.mean():.4f}" if len(act) > 0 else "—"
        c_str = f"{cor.mean():.4f}" if len(cor) > 0 else "—"
        desc_lines.append(f"| {v} | {n} | {a_str} | {c_str} |")
    desc_lines.append("")

    # Holm summary table
    holm_lines = ["## Holm-Corrected Contrasts (vs frankenstein)", ""]
    holm_lines.append("| Contrast | Metric | delta | Wald p (raw) | Holm-adjusted p |")
    holm_lines.append("|----------|--------|-------|--------------|-----------------|")
    for key in sorted(adj_p):
        metric_name, variant_name = key.split(":", 1)
        rp = raw_p[key]; ap = adj_p[key]; c = coefs[key]
        rp_str = "< .001" if rp < 0.001 else f"{rp:.3f}"
        ap_str = "< .001" if ap < 0.001 else f"{ap:.3f}"
        holm_lines.append(f"| {variant_name} | {metric_name} | {c:+.4f} | {rp_str} | {ap_str} |")
    holm_lines.append("")

    report = [
        "# Phase 2b — Prompt Variant Validation (frankenstein vs originals)",
        "",
        "Model: `score ~ C(variant) + (1|q_num)`",
        f"Reference level: `{REF_VARIANT}` (current production prompt)",
        f"Judge: `{JUDGE_MODEL}` via OpenRouter",
        "Test set: 45 corpus-grounded synthetic goldens (goldens_synth_v1.json)",
        "RAG: on (hybrid 70/30, k=4). Sensors: off.",
        "Family-wise correction: Holm across all non-reference contrasts (2 metrics x 3 variants = 6 tests).",
        "",
    ]
    report += desc_lines
    report.append("## LME Fixed Effects")
    report.append("")
    report += fe_blocks
    report += holm_lines

    return "\n".join(report) + "\n"


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Phase 2b prompt variant validation")
    parser.add_argument("--n", type=int, default=None, help="Limit to first N questions (smoke test)")
    parser.add_argument("--k", type=int, default=4)
    parser.add_argument("--model", type=str, default=None, help="LLM model override")
    args = parser.parse_args()

    Path(RESULTS_DIR).mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("PHASE 2b — PROMPT VARIANT VALIDATION")
    print(f"  Variants:  {', '.join(PROMPT_VARIANTS)}")
    print(f"  Judge:     {JUDGE_MODEL}")
    print(f"  Goldens:   {GOLDENS_PATH}")
    print(f"  k={args.k}  hybrid=True  weights=[0.7, 0.3]")
    print("=" * 80)

    print("\nLoading goldens...")
    questions = load_goldens()
    if args.n:
        questions = questions[:args.n]
    print(f"  {len(questions)} questions loaded")

    print("\nSetting up judge...")
    judge = OpenRouterJudge()

    actionability_metric = GEval(
        name="Actionability",
        criteria=ACTIONABILITY_CRITERIA,
        evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
        model=judge,
        rubric=_ACTION_RUBRIC,
        async_mode=False,
    )
    correctness_metric = GEval(
        name="Correctness",
        criteria=CORRECTNESS_CRITERIA,
        evaluation_params=[
            LLMTestCaseParams.INPUT,
            LLMTestCaseParams.ACTUAL_OUTPUT,
            LLMTestCaseParams.EXPECTED_OUTPUT,
        ],
        model=judge,
        rubric=_CORRECT_RUBRIC,
        async_mode=False,
    )

    print("\nBuilding RAG pipeline...")
    KB_PATH    = os.path.join(_ROOT, "test_docs")
    CHROMA_DIR = os.path.join(_ROOT, "chroma_db")
    docs    = load_documents(KB_PATH)
    chunks  = split_documents(docs, chunk_size=1000)
    vectordb = build_vector_store(chunks, persist_dir=CHROMA_DIR, folder_path=KB_PATH)
    bm25    = build_bm25_retriever(chunks)
    print("RAG pipeline ready.\n")

    all_results: Dict[str, List[Dict]] = {}

    for variant_name, prompt_template in PROMPT_VARIANTS.items():
        print(f"\n{'─' * 60}")
        print(f"Variant: {variant_name.upper()}  ({len(questions)} questions)")
        print(f"{'─' * 60}")
        all_results[variant_name] = run_variant(
            variant_name=variant_name,
            prompt_template=prompt_template,
            questions=questions,
            vectordb=vectordb,
            bm25_retriever=bm25,
            actionability_metric=actionability_metric,
            correctness_metric=correctness_metric,
            k=args.k,
            llm_model=args.model,
        )

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY  (G-Eval, 0-1 scale, -1 = judge failure excluded)")
    print("=" * 80)
    print(f"{'Variant':<15} {'Actionability':>14} {'Correctness':>12} {'Failures':>10}")
    print("-" * 55)
    for variant_name, results in all_results.items():
        valid_a = [r["actionability"] for r in results if r["actionability"] >= 0]
        valid_c = [r["correctness"]   for r in results if r["correctness"]   >= 0]
        fails   = sum(1 for r in results if r["actionability"] < 0 or r["correctness"] < 0)
        avg_a = sum(valid_a) / len(valid_a) if valid_a else float("nan")
        avg_c = sum(valid_c) / len(valid_c) if valid_c else float("nan")
        print(f"{variant_name:<15} {avg_a:>14.4f} {avg_c:>12.4f} {fails:>10}")

    # Save raw results
    payload = {
        "judge_model":  JUDGE_MODEL,
        "goldens_file": GOLDENS_PATH,
        "n_questions":  len(questions),
        "conditions":   all_results,
    }
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(f"\nRaw results -> {OUT_JSON}")

    # Run LME and save report
    print("\nRunning LME analysis...")
    report = _run_lme(all_results)
    with open(OUT_MD, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"LME report  -> {OUT_MD}")


if __name__ == "__main__":
    main()
