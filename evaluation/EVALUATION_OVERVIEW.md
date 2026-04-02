# Evaluation Pipeline Overview

<!-- AUTO-GENERATED -->
## Current Status (as of 2026-04-01)

| Phase | Component | Status | Output |
|-------|-----------|--------|--------|
| Phase 1 — Configuration sweep | `sweep.py` (24 configs × 30 questions) | **Complete** | `results/sweep_*.csv` |
| Phase 1 — Statistical analysis | `sweep_analysis.py`, `sweep_mixed_model.py` | **Complete** | `results/mixed_model_analysis.md` |
| Phase 2 — Prompt variant scoring (G-Eval) | `evaluate_prompt_variants.py` | **Complete** | `results/prompt_variant_results.json` |
| Phase 2 — Prompt variant mixed model | `prompt_variant_mixed_model.py` | **Complete** | `results/prompt_variant_mixed_model.md` |
| Phase 2 — Human pairwise ranking | `ranking_app/` (37 raters, 644 votes) | **Complete** | Supabase `rankings` table |
| Phase 2 — Sensor routing comparison | `compare_sensor_routing.py` (19 scenarios × 3 runs) | **Complete** | `results/sensor_routing_comparison.json` |
| Phase 2 — Sensor awareness end-to-end | `evaluate_sensor_awareness.py` (19 scenarios) | **Complete** | `results/sensor_awareness_results.json` |

### Phase 1 — Winner Configuration

**Production config** (from sweep winner): `mistralai/ministral-14b-2512`, chunk=1000, k=4, weights=[0.7, 0.3] (70% semantic / 30% BM25).

LLM model is the only statistically significant factor (ANOVA F=34.85, p<0.0001). Chunk size and k are significant for retrieval quality but not for answer quality at this scale. Mixed model analysis with random intercept per question confirms: k=4 and chunk=1000 both have positive, significant effects when question-level variance is removed.

### Phase 2 — Key Results

**Prompt variant winner: `structured`** — 56.1% human win rate (37 raters, 644 votes), best actionability (0.890 vs 0.827 baseline, p<0.001). G-Eval judge: Kimi 2.6. Mixed effects model with random intercept per question confirms significance after Bonferroni correction. New hybrid production prompts combine structured layout + conciseness + sensor data as specificity amplifier.

**Sensor routing winner: LLM classifier** — 19/19 (100%) accuracy vs keyword filter's 18/19 (94.7%). Cost: ~$0.00009/call. Now production default (`SENSOR_ROUTING_MODE=llm`). Keyword filter fails on S13 (H₂S critical + encyclopedic question) due to rigid rule priority ordering.

**Sensor awareness pass rate:** 16/19 (84.2%). Three failure modes: over-hedging on normal readings (S03), urgency miscalibration on encyclopedic questions (S11), H₂S critical overriding encyclopedic exclusion rule (S13 — now fixed by LLM routing).
<!-- END AUTO-GENERATED -->

---

## Scripts — What Each One Tests

| Script | Phase | Purpose | Metrics | Judge | When to Run |
|--------|-------|---------|---------|-------|-------------|
| `evaluate_rag.py` | 1 | RAG vs no-RAG (fast heuristic) | Topic coverage, length, actionability (keyword) | None (no API) | Quick baseline, zero cost |
| `evaluate_retrieval.py` | 1 | Hybrid vs semantic retrieval | Heuristic + RAGAS ContextPrecision + latency | Claude Haiku | Before sweep (retrieval method comparison) |
| `evaluate_ragas.py` | 1 | RAG vs no-RAG (semantic) | Faithfulness, Answer Relevancy, Context Precision, Context Recall | Claude Haiku | Standalone RAG quality check |
| `evaluate_deepeval.py` | 1 | RAG vs no-RAG (custom + standard) | Actionability, Correctness (GEval) + Faithfulness, Answer Relevancy | Claude Haiku | After sweep winner is known |
| `sweep.py` | 1 | Hyperparameter sweep (full factorial) | All 7 metrics below | Llama 3.3 70B (OpenRouter) | Round 1 experiment |
| `sweep_analysis.py` | 1 | Statistical analysis of sweep results | ANOVA, main effects, interactions, composite | N/A (post-processing) | After sweep completes |
| `sweep_mixed_model.py` | 1 | Linear mixed effects analysis | Fixed effects + BLUPs per question | N/A (post-processing) | After sweep completes |
| `evaluate_prompt_variants.py` | 2 | Prompt variant scoring + pair export | Actionability, Correctness (GEval) | Kimi 2.6 (OpenRouter) | After sweep winner config is set |
| `prompt_variant_mixed_model.py` | 2 | Mixed effects analysis of prompt variants | Fixed effects + pairwise contrasts + BLUPs | N/A (post-processing) | After prompt variant scoring |
| `compare_sensor_routing.py` | 2 | Keyword vs LLM routing comparison | Accuracy, latency, cost per call | N/A (direct comparison) | After sensor_filter.py changes |
| `evaluate_sensor_awareness.py` | 2 | Sensor awareness end-to-end (19 scenarios) | Routing, value citation, urgency, hallucination | N/A (rule-based checks) | After sensor pipeline changes |
| `run_eval.py` | — | Orchestrator — runs all evaluate_*.py scripts | All of the above | Mixed | Full evaluation run |

## Metrics — What Each One Measures

| Metric | Type | Scale | What It Measures | Failure Mode It Catches |
|--------|------|-------|-----------------|------------------------|
| **Actionability** | Custom GEval | 0–1 | Does the keeper know what to do after reading? | Vague, hedged, or filler answers |
| **Correctness** | Custom GEval | 0–1 | Is the advice factually accurate and safe? | Wrong temps, dangerous meds, bad feeding info |
| **Faithfulness** | Standard (DeepEval) | 0–1 | Is the answer grounded in retrieved context? | Hallucination — model invents facts not in docs |
| **Answer Relevancy** | Standard (DeepEval) | 0–1 | Does the answer address the actual question? | Off-topic or tangential responses |
| **Contextual Precision** | Standard (DeepEval) | 0–1 | Are relevant chunks ranked above irrelevant ones? | Retriever returns noise at top positions |
| **Contextual Recall** | Standard (DeepEval) | 0–1 | Does retrieved context contain needed information? | Missing chunks — answer can't be derived from context |
| **Contextual Relevancy** | Standard (DeepEval) | 0–1 | Are retrieved chunks actually relevant to the query? | Retriever returns too many unrelated chunks |

## GEval Rubric Mapping

The custom GEval metrics (Actionability, Correctness) use a 3-tier rubric on the 0–10 scale:

| Score Range | Level | Meaning |
|-------------|-------|---------|
| 0–3 | Bad | Not actionable / Incorrect or harmful |
| 4–6 | Mediocre | Somewhat useful but gaps / Mostly correct but weaknesses |
| 7–10 | Good | Genuinely helpful / Correct and appropriate |

DeepEval normalizes: `final_score = raw_score / 10` (so 0.0–1.0).

Criteria text is in `eval_config.py`. Rubric objects are in `sweep.py` and `evaluate_deepeval.py`.

## Required LLMTestCase Fields Per Metric

| Metric | input | actual_output | expected_output | retrieval_context |
|--------|:-----:|:-------------:|:---------------:|:-----------------:|
| Actionability (GEval) | x | x | | |
| Correctness (GEval) | x | x | | |
| Faithfulness | x | x | | x |
| Answer Relevancy | x | x | | |
| Contextual Precision | x | | x | x |
| Contextual Recall | x | | x | x |
| Contextual Relevancy | x | | | x |

## Score Handling

- **-1 sentinel**: Returned when the judge fails to produce parseable JSON. Excluded from all averages, composites, ANOVA, and main effects.
- **Bounds check**: `_measure()` rejects scores outside 0–1 range (returns -1).
- **Fail rate tracking**: Per-metric failure rates are stored in results; warns if >20% of questions fail for any metric.
- **Composite score**: Mean of all 7 avg metrics, excluding -1 sentinels (missing-at-random treatment).
