# Mixed Effects Analysis — Retrieved vs Random Chunks Ablation

**Research question:** Does targeted retrieval (hybrid 70/30 BM25+semantic) find meaningfully better context than chance, measured by downstream answer quality?

Model: `score ~ C(condition) + (1|q_num)` (random intercept per question removes between-question difficulty noise from the condition effect.)
Reference level: condition=`random_chunks`
The coefficient `condition=rag` is the estimated mean gain on the 0–1 metric scale.
Family-wise correction: Holm across all five metrics.

**Key metric:** `contextual_recall` — directly measures whether the retrieved context contains the information needed to answer each question. Random chunks → near zero (corpus is large, relevant chunks rare by chance). Retrieved chunks → high (retriever selects relevant material). The gap is the core empirical claim.

## Descriptive Summary

All five metrics apply to both conditions (both have retrieval_context).

| Condition | N | Answer Relevancy | Actionability | Correctness | Faithfulness | Contextual Recall |
|-----------|---|-----|-----|-----|-----|-----|
| random_chunks | 45 | 0.8020 | 0.9289 | 0.5644 | 0.9951 | 0.0450 |
| rag | 45 | 0.8369 | 0.9244 | 0.6044 | 0.9681 | 0.9519 |

**Faithfulness interpretation note:** For `random_chunks`, high faithfulness indicates the model produced claims consistent with irrelevant context (confabulation from noise). Low faithfulness means the model ignored irrelevant context and drew on its own knowledge. For `rag`, high faithfulness is the desired outcome — the model stays grounded in relevant retrieved material.

## LME Results — All Metrics

### Answer Relevancy

Random effects variance (σ²_q): 0.025796
N observations: 90

| Parameter | Coef (Δ) | SE | z | p (Wald) | 95% CI |
|-----------|----------|----|---|----------|--------|
| Intercept (random_chunks) | 0.8020 | 0.0400 | +20.03 | < .001 | [+0.7235, +0.8805] |
| condition=rag | +0.0349 | 0.0454 | +0.77 | 0.442 | [-0.0540, +0.1239] |

### Actionability

Random effects variance (σ²_q): 0.007005
N observations: 90

| Parameter | Coef (Δ) | SE | z | p (Wald) | 95% CI |
|-----------|----------|----|---|----------|--------|
| Intercept (random_chunks) | 0.9289 | 0.0147 | +63.20 | < .001 | [+0.9001, +0.9577] |
| condition=rag | -0.0044 | 0.0110 | -0.40 | 0.686 | [-0.0260, +0.0171] |

### Correctness

Random effects variance (σ²_q): 0.006071
N observations: 90

| Parameter | Coef (Δ) | SE | z | p (Wald) | 95% CI |
|-----------|----------|----|---|----------|--------|
| Intercept (random_chunks) | 0.5644 | 0.0161 | +35.13 | < .001 | [+0.5330, +0.5959] |
| condition=rag | +0.0400 | 0.0157 | +2.55 | 0.011 | [+0.0092, +0.0708] |

### Faithfulness

Random effects variance (σ²_q): 0.000000
N observations: 90

| Parameter | Coef (Δ) | SE | z | p (Wald) | 95% CI |
|-----------|----------|----|---|----------|--------|
| Intercept (random_chunks) | 0.9951 | 0.0069 | +143.70 | < .001 | [+0.9815, +1.0087] |
| condition=rag | -0.0270 | 0.0098 | -2.76 | 0.006 | [-0.0462, -0.0078] |

### Contextual Recall

Random effects variance (σ²_q): 0.000000
N observations: 90

| Parameter | Coef (Δ) | SE | z | p (Wald) | 95% CI |
|-----------|----------|----|---|----------|--------|
| Intercept (random_chunks) | 0.0450 | 0.0274 | +1.64 | 0.101 | [-0.0088, +0.0988] |
| condition=rag | +0.9069 | 0.0388 | +23.37 | < .001 | [+0.8308, +0.9829] |

### Holm-corrected family-wise p-values

| Metric | Δ (rag − random_chunks) | Wald p (raw) | Holm-adjusted p |
|--------|-------------------------|--------------|-----------------|
| answer_relevancy | +0.0349 | 0.442 | 0.883 |
| actionability | -0.0044 | 0.686 | 0.883 |
| correctness | +0.0400 | 0.011 | 0.033 |
| faithfulness | -0.0270 | 0.006 | 0.023 |
| contextual_recall | +0.9069 | < .001 | < .001 |

**Faithfulness interpretation note:** For `random_chunks`, high faithfulness indicates the model produced claims consistent with irrelevant context (confabulation from noise). Low faithfulness means the model ignored irrelevant context and drew on its own knowledge. For `rag`, high faithfulness is the desired outcome — the model stays grounded in relevant retrieved material.

---
## Bad Question Detection

### Systematically Hard Questions (BLUPs)

Random intercept stats (anchored on correctness LME): mean=0.000, SD=0.064, cutoff (< mean − 1.5×SD) = -0.096

| Q# | BLUP | Question |
|----|------|---------|
| 26 | -0.264 | How many chickens should I keep in one flock? |
| 5 | -0.161 | How much space do chickens need? |
| 40 | -0.127 | Q40 |

## Robustness Re-run (flagged questions dropped)

Dropping Q#s [5, 26, 40] and refitting all metrics. Checks whether the main effect is driven by systematically hard questions.

| Metric | Δ (rag − random_chunks) | Wald p (raw) | Holm-adjusted p |
|--------|-------------------------|--------------|-----------------|
| answer_relevancy | +0.0410 | 0.393 | 0.785 |
| actionability | -0.0071 | 0.535 | 0.785 |
| correctness | +0.0262 | 0.040 | 0.121 |
| faithfulness | -0.0261 | 0.011 | 0.043 |
| contextual_recall | +0.9002 | < .001 | < .001 |

