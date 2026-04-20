# Evaluation Pipeline Overview

<!-- AUTO-GENERATED -->
## Current Status (as of 2026-04-19)

| Phase | Component | Status | Output |
|-------|-----------|--------|--------|
| Phase 1 — Configuration sweep | `sweep.py` (24 configs × 30 questions) | **Complete** | `results/sweep_*.csv` |
| Phase 1 — Statistical analysis | `sweep_analysis.py`, `sweep_mixed_model.py` | **Complete** | `results/mixed_model_analysis.md` |
| Phase 2 — Prompt variant scoring (G-Eval) | `evaluate_prompt_variants.py` | **Complete** | `results/prompt_variant_results.json` |
| Phase 2 — Prompt variant mixed model | `prompt_variant_mixed_model.py` | **Complete** | `results/prompt_variant_mixed_model.md` |
| Phase 2 — Human pairwise ranking | `ranking_app/` (37 raters, 644 votes) | **Complete** | Supabase `rankings` table |
| Phase 2 — Sensor routing comparison | `compare_sensor_routing.py` (19 scenarios × 3 runs) | **Complete** | `results/sensor_routing_comparison.json` |
| Phase 2 — Sensor awareness end-to-end | `evaluate_sensor_awareness.py` (19 scenarios) | **Complete** | `results/sensor_awareness_results.json` |
| Appendix — RAG vs no-RAG ablation | `evaluate_rag_ablation.py` + `rag_ablation_mixed_model.py` (2 conditions × 45 synthetic goldens) | **Complete** | `results/rag_ablation_results.json`, `results/rag_ablation_mixed_model.md` |

### Phase 1 — Winner Configuration

**Production config** (from sweep winner): `mistralai/ministral-14b-2512`, chunk=1000, k=4, weights=[0.7, 0.3] (70% semantic / 30% BM25).

LLM model is the only statistically significant factor (ANOVA F=34.85, p<0.0001). Chunk size and k are significant for retrieval quality but not for answer quality at this scale. Mixed model analysis with random intercept per question confirms: k=4 and chunk=1000 both have positive, significant effects when question-level variance is removed.

### Phase 2 — Key Results

**Prompt variant winner: `structured`** — 56.1% human win rate (37 raters, 644 votes), best actionability (0.890 vs 0.827 baseline, p<0.001). G-Eval judge: Kimi 2.6. Mixed effects model with random intercept per question confirms significance after Bonferroni correction. New hybrid production prompts combine structured layout + conciseness + sensor data as specificity amplifier.

**Sensor routing winner: LLM classifier** — 19/19 (100%) accuracy vs keyword filter's 18/19 (94.7%). Cost: ~$0.00009/call. Now production default (`SENSOR_ROUTING_MODE=llm`). Keyword filter fails on S13 (H₂S critical + encyclopedic question) due to rigid rule priority ordering.

**Sensor awareness pass rate:** 16/19 (84.2%). Three failure modes: over-hedging on normal readings (S03), urgency miscalibration on encyclopedic questions (S11), H₂S critical overriding encyclopedic exclusion rule (S13 — now fixed by LLM routing).

### Appendix — RAG vs no-RAG Ablation Results

**Metric stack** (refactored): AnswerRelevancyMetric + FaithfulnessMetric + ContextualRecallMetric (DeepEval built-ins) + Actionability + Correctness (custom G-Eval). Judge: `gpt-4o-mini`. Test set: 45 synthetic goldens from `test_docs/` (DeepEval Synthesizer, gpt-4o-mini, 5 evolution types).

**Shared metrics (LME, Holm-corrected):**

| Metric | no_rag | rag | Δ | Holm p |
|--------|--------|-----|---|--------|
| Answer Relevancy | 0.795 | 0.827 | +0.032 | 1.000 |
| Actionability | 0.927 | 0.933 | +0.007 | 1.000 |
| Correctness | 0.582 | 0.602 | +0.020 | 0.367 |

**RAG-only diagnostics:** Faithfulness = 0.961 (SD=0.058), Contextual Recall = 0.945 (SD=0.187).

**Interpretation:** RAG shows small positive Δ across all metrics but none reach significance. The retriever quality is excellent (faithfulness ~0.96), but the base LLM already has sufficient parametric knowledge for common poultry husbandry Q&A. Two systematically hard questions flagged by BLUPs (Q26: flock size, Q27: laying age); robustness re-run confirms main results hold after dropping them.
<!-- END AUTO-GENERATED -->

---

## Scripts — What Each One Tests

| Script | Phase | Purpose | Metrics | Judge | When to Run |
|--------|-------|---------|---------|-------|-------------|
| `sweep.py` | 1 | Hyperparameter sweep (full factorial) | Actionability, Correctness, Faithfulness, Answer Relevancy, Contextual Recall | OpenRouter (via `SWEEP_JUDGE_MODEL`) | Round 1 experiment |
| `sweep_analysis.py` | 1 | Statistical analysis of sweep results | ANOVA, main effects, interactions, composite | N/A (post-processing) | After sweep completes |
| `sweep_mixed_model.py` | 1 | Linear mixed effects analysis | Fixed effects + BLUPs per question | N/A (post-processing) | After sweep completes |
| `evaluate_prompt_variants.py` | 2 | Prompt variant scoring + pair export | Actionability, Correctness (G-Eval) | OpenRouter (via `SWEEP_JUDGE_MODEL`) | After sweep winner config is set |
| `prompt_variant_mixed_model.py` | 2 | Mixed effects analysis of prompt variants | Fixed effects + pairwise contrasts + BLUPs | N/A (post-processing) | After prompt variant scoring |
| `compare_sensor_routing.py` | 2 | Keyword vs LLM routing comparison | Accuracy, latency, cost per call | N/A (direct comparison) | After `sensor_filter.py` changes |
| `evaluate_sensor_awareness.py` | 2 | Sensor awareness end-to-end (19 scenarios) | Routing, value citation, urgency, hallucination | N/A (rule-based checks) | After sensor pipeline changes |
| `evaluate_rag_ablation.py` | Appendix | RAG vs no-RAG (2 conditions, 45 synthetic goldens, max_tokens=1000) | Answer Relevancy, Actionability, Correctness (shared); Faithfulness, Contextual Recall (rag-only) | OpenRouter (via `SWEEP_JUDGE_MODEL`, gpt-4o-mini) | Paper appendix — isolates retrieval contribution |
| `rag_ablation_mixed_model.py` | Appendix | LME for ablation (score ~ condition + (1|q_num)) | Fixed-effect Δ, Wald p, Holm-adjusted, BLUPs | N/A (post-processing) | After `evaluate_rag_ablation.py` |

## Judge Model — Per-Experiment Record

> **Methodological note:** Different LLM judges were used across evaluation phases. Judge calibration, strictness, and JSON compliance differ between models. **Do not compare raw metric means across phases** — comparisons are only valid within a single script run (e.g. `no_rag` vs `rag` within the ablation, or prompt variants against each other within Phase 2).

| Phase / Script | Judge Model | Notes |
|----------------|-------------|-------|
| Phase 1 — `sweep.py` | `meta-llama/llama-3.3-70b-instruct` via OpenRouter | SWEEP_JUDGE_MODEL at time of run |
| Phase 2 — `evaluate_prompt_variants.py` | `moonshotai/kimi-k2` (Kimi 2.6) via OpenRouter | SWEEP_JUDGE_MODEL at time of run |
| Phase 2 — Human ranking | Human raters (37 raters, 644 votes) | No LLM judge |
| Phase 2 — Sensor awareness | Rule-based checks | No LLM judge |
| Appendix — `evaluate_rag_ablation.py` | `openai/gpt-4o-mini` via OpenRouter | Switched from minimax/minimax-m2.7 which failed DeepEval's built-in metric JSON schema |

## Metrics — What Each One Measures

| Metric | Type | Scale | What It Measures | Failure Mode It Catches |
|--------|------|-------|-----------------|------------------------|
| **Actionability** | Custom G-Eval | 0–1 | Does the keeper know what to do after reading? | Vague, hedged, or filler answers |
| **Correctness** | Custom G-Eval | 0–1 | Is the advice factually accurate and safe? | Wrong temps, dangerous meds, bad feeding info |
| **Faithfulness** | Standard DeepEval built-in (sweep + ablation) | 0–1 | Is the answer grounded in retrieved context? | Hallucination — model invents facts not in docs |
| **Answer Relevancy** | Standard DeepEval built-in (sweep + ablation) | 0–1 | Does the answer address the actual question? | Off-topic or tangential responses |
| **Contextual Recall** | Standard DeepEval built-in (sweep + ablation) | 0–1 | Does retrieved context contain needed information? | Missing chunks — answer can't be derived from context |

## GEval Rubric Mapping

**Scale ownership.** Rubric objects (`Rubric(score_range=...)`) in each script define the numeric scoring scale. Criteria text in `eval_config.py` describes the qualitative tiers only — it intentionally does not reference numeric scores. DeepEval normalizes: `final_score = raw_score / 10` (so 0.0–1.0).

**Per-script rubrics** (raw 0–10 scale):

| Script | Rubric shape | Tiers |
|---|---|---|
| `sweep.py` (Actionability, Correctness) | 3-tier | `0–3` Bad · `4–6` Mediocre · `7–10` Good |
| `evaluate_prompt_variants.py` (Actionability, Correctness) | 4-tier | `0–3` Bad · `4–5` Partial · `6–7` Good · `8–10` Fully good |
| `evaluate_rag_ablation.py` (Actionability, Correctness) | 4-tier | `0–3` Bad · `4–5` Partial · `6–7` Good · `8–10` Fully good |

Because rubric shapes differ across scripts, do not compare raw means across scripts — compare within a script (e.g. `no_rag` vs `rag` in the ablation, or prompt variants against each other).

**Files:**
- Criteria text — `eval_config.py` (`ACTIONABILITY_CRITERIA`, `CORRECTNESS_CRITERIA`).
- Rubric objects — `sweep.py:486-493`, `evaluate_prompt_variants.py:97-108`, `evaluate_rag_ablation.py` (GEval rubrics).

## Required LLMTestCase Fields Per Metric

| Metric | input | actual_output | expected_output | retrieval_context |
|--------|:-----:|:-------------:|:---------------:|:-----------------:|
| Actionability (G-Eval) | x | x | | |
| Correctness (G-Eval) | x | x | x | |
| Faithfulness | x | x | | x |
| Answer Relevancy | x | x | | |
| Contextual Recall | x | | x | x |

## Score Handling

- **-1 sentinel**: Returned when the judge fails to produce parseable JSON. Excluded from all averages, composites, ANOVA, and main effects.
- **Bounds check**: `_measure()` rejects scores outside 0–1 range (returns -1).
- **Fail rate tracking**: Per-metric failure rates are stored in results; warns if >20% of questions fail for any metric.
- **Composite score**: Mean of all 7 avg metrics, excluding -1 sentinels (missing-at-random treatment).
