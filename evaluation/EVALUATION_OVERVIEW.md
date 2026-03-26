# Evaluation Pipeline Overview

<!-- AUTO-GENERATED -->
## Current Status (as of 2026-03-22)

| Phase | Component | Status |
|-------|-----------|--------|
| Phase 1 — Configuration sweep | `sweep.py` (24 configs × 30 questions) | **Complete** |
| Phase 1 — Statistical analysis | `sweep_analysis.py`, `sweep_mixed_model.py` | **Complete** |
| Phase 2 — Prompt variant scoring | `evaluate_prompt_variants.py` (G-Eval) | **Complete** |
| Phase 2 — Human pairwise ranking | `human_ranking.py` | Pending |
| Phase 2 — Sensor awareness | `evaluate_sensor_awareness.py` (19 scenarios) | **Complete** |

**Production configuration** (from sweep winner): `mistralai/ministral-14b-2512`, chunk=1000, k=4, weights=70/30 (hybrid BM25/semantic).

**Sensor awareness pass rate:** 16/19 (84.2%). Three scenarios fail: over-hedging on normal readings (S03), urgency miscalibration on encyclopedic questions (S11), and sensor context leaking into encyclopedic answers under H₂S-critical conditions (S13).
<!-- END AUTO-GENERATED -->

---

## Scripts — What Each One Tests

| Script | Purpose | Metrics | Judge | When to Run |
|--------|---------|---------|-------|-------------|
| `evaluate_rag.py` | RAG vs no-RAG (fast heuristic) | Topic coverage, length, actionability (keyword) | None (no API) | Quick baseline, zero cost |
| `evaluate_retrieval.py` | Hybrid vs semantic retrieval | Heuristic + RAGAS ContextPrecision + latency | Claude Haiku | Before sweep (retrieval method comparison) |
| `evaluate_ragas.py` | RAG vs no-RAG (semantic) | Faithfulness, Answer Relevancy, Context Precision, Context Recall | Claude Haiku | Standalone RAG quality check |
| `evaluate_deepeval.py` | RAG vs no-RAG (custom + standard) | Actionability, Correctness (GEval) + Faithfulness, Answer Relevancy | Claude Haiku | After sweep winner is known |
| `sweep.py` | Hyperparameter sweep (full factorial) | All 7 metrics below | Llama 3.3 70B (OpenRouter) | Round 1 experiment |
| `sweep_analysis.py` | Statistical analysis of sweep results | ANOVA, main effects, interactions, composite | N/A (post-processing) | After sweep completes |
| `run_eval.py` | Orchestrator — runs all evaluate_*.py scripts | All of the above | Mixed | Full evaluation run |

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
