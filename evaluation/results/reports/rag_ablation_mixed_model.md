# Mixed Effects Analysis — RAG vs no-RAG Ablation

Model: `score ~ C(condition) + (1|q_num)` (shared metrics only)
Random intercept per question removes between-question noise from the condition effect.
Reference level: condition=`no_rag`

With two conditions, the non-intercept coefficient `condition=rag` is the estimated
mean difference in score on the 0–1 metric scale and is the reported effect size.
Family-wise correction: Holm across the three shared metrics.

The rag-only metrics (`faithfulness`, `contextual_recall`) require retrieval_context
so have no no_rag counterpart — they are reported as descriptive diagnostics.

## Descriptive Summary

Shared metrics (both conditions) and RAG-only diagnostics.

| Condition | N | Answer Relevancy | Actionability | Correctness | Faithfulness | Contextual Recall |
|-----------|---|-----|-----|-----|-----|-----|
| no_rag | 45 | 0.7954 | 0.9267 | 0.5822 | — | — |
| rag | 45 | 0.8274 | 0.9333 | 0.6022 | 0.9606 | 0.9449 |

## Shared Metrics — LME (full sample)

### Answer Relevancy

Random effects variance (σ²_q): 0.011073
N observations: 90

| Parameter | Coef (Δ) | SE | z | p (Wald) | 95% CI |
|-----------|----------|----|---|----------|--------|
| Intercept (no_rag) | 0.7954 | 0.0432 | +18.39 | < .001 | [+0.7107, +0.8802] |
| condition=rag | +0.0320 | 0.0570 | +0.56 | 0.575 | [-0.0797, +0.1437] |

### Actionability

Random effects variance (σ²_q): 0.009318
N observations: 90

| Parameter | Coef (Δ) | SE | z | p (Wald) | 95% CI |
|-----------|----------|----|---|----------|--------|
| Intercept (no_rag) | 0.9267 | 0.0164 | +56.43 | < .001 | [+0.8945, +0.9589] |
| condition=rag | +0.0067 | 0.0112 | +0.60 | 0.551 | [-0.0153, +0.0286] |

### Correctness

Random effects variance (σ²_q): 0.012313
N observations: 90

| Parameter | Coef (Δ) | SE | z | p (Wald) | 95% CI |
|-----------|----------|----|---|----------|--------|
| Intercept (no_rag) | 0.5822 | 0.0189 | +30.79 | < .001 | [+0.5452, +0.6193] |
| condition=rag | +0.0200 | 0.0129 | +1.54 | 0.122 | [-0.0054, +0.0454] |

### Holm-corrected family-wise p-values

| Metric | Δ (rag − no_rag) | Wald p (raw) | Holm-adjusted p |
|--------|------------------|--------------|-----------------|
| answer_relevancy | +0.0320 | 0.575 | 1.000 |
| actionability | +0.0067 | 0.551 | 1.000 |
| correctness | +0.0200 | 0.122 | 0.367 |

---
## RAG-only Retrieval Diagnostics

These metrics require `retrieval_context` so they are only defined for the rag condition.
No inferential test — these are descriptive quality signals for the retriever + generator.

| Metric | N | Mean | SD | Min | Max |
|--------|---|------|----|-----|-----|
| faithfulness | 45 | 0.9606 | 0.0581 | 0.7500 | 1.0000 |
| contextual_recall | 45 | 0.9449 | 0.1871 | 0.0000 | 1.0000 |

---
## Bad Question Detection

### Systematically Hard Questions (BLUPs)

Random intercept stats (anchored on correctness LME): mean=0.000, SD=0.102, cutoff (< mean − 1.5×SD) = -0.153

| Q# | BLUP | Question |
|----|------|---------|
| 26 | -0.514 | How many chickens should I keep in one flock? |
| 27 | -0.167 | When do chickens start laying eggs? |

## Robustness Re-run (flagged questions dropped)

Dropping Q#s [26, 27] and refitting the three shared metrics. This checks whether the main effect is driven by a handful of systematically hard questions.

| Metric | Δ (rag − no_rag) | Wald p (raw) | Holm-adjusted p |
|--------|------------------|--------------|-----------------|
| answer_relevancy | +0.0269 | 0.649 | 1.000 |
| actionability | +0.0047 | 0.686 | 1.000 |
| correctness | +0.0209 | 0.122 | 0.367 |

