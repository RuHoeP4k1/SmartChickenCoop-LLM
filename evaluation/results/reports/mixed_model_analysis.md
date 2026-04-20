# Mixed Effects Analysis — RAG Sweep

Models: `score ~ C(llm) + C(chunk) + C(k) + C(weights) + (1|q_num)`
Random intercept per question removes between-question noise from fixed effect estimates.
Reference levels: llm=qwen3-8b, chunk=600, k=2, weights=pure-sem

Significance: \* p<0.05  \*\* p<0.01  (pairwise contrasts: Bonferroni corrected)

## Individual Metrics

### Actionability

Random effects variance (σ²_q): 0.001447
N observations: 719

| Parameter | Coef | SE | z | p | 95% CI | Sig |
|-----------|------|----|---|---|--------|-----|
| llm=ministral-14b | +0.0488 | 0.0043 | +11.40 | < .001 | [+0.0404, +0.0571] | ** |
| llm=mistral-small-24b | -0.0113 | 0.0043 | -2.63 | 0.009 | [-0.0196, -0.0029] | ** |
| chunk=1000 | +0.0139 | 0.0035 | +3.97 | < .001 | [+0.0070, +0.0207] | ** |
| k=4 | +0.0144 | 0.0035 | +4.13 | < .001 | [+0.0076, +0.0213] | ** |
| weights=70/30 | +0.0056 | 0.0035 | +1.59 | 0.112 | [-0.0013, +0.0124] |  |

**Pairwise LLM contrasts (actionability, Bonferroni corrected)**

| Comparison | Diff | SE | z | p (adj) | 95% CI | Sig |
|-----------|------|----|---|---------|--------|-----|
| ministral-14b vs qwen3-8b | +0.0488 | 0.0043 | +11.40 | < .001 | [+0.0404, +0.0571] | ** |
| mistral-small-24b vs qwen3-8b | -0.0113 | 0.0043 | -2.63 | 0.026 | [-0.0196, -0.0029] | * |
| mistral-small-24b vs ministral-14b | -0.0600 | 0.0043 | -14.01 | < .001 | [-0.0684, -0.0516] | ** |

### Correctness

Random effects variance (σ²_q): 0.000951
N observations: 720

| Parameter | Coef | SE | z | p | 95% CI | Sig |
|-----------|------|----|---|---|--------|-----|
| llm=ministral-14b | +0.0383 | 0.0045 | +8.49 | < .001 | [+0.0295, +0.0472] | ** |
| llm=mistral-small-24b | +0.0137 | 0.0045 | +3.05 | 0.002 | [+0.0049, +0.0226] | ** |
| chunk=1000 | +0.0044 | 0.0037 | +1.21 | 0.228 | [-0.0028, +0.0117] |  |
| k=4 | +0.0078 | 0.0037 | +2.11 | 0.035 | [+0.0006, +0.0150] | * |
| weights=70/30 | +0.0017 | 0.0037 | +0.45 | 0.651 | [-0.0056, +0.0089] |  |

**Pairwise LLM contrasts (correctness, Bonferroni corrected)**

| Comparison | Diff | SE | z | p (adj) | 95% CI | Sig |
|-----------|------|----|---|---------|--------|-----|
| ministral-14b vs qwen3-8b | +0.0383 | 0.0045 | +8.49 | < .001 | [+0.0295, +0.0472] | ** |
| mistral-small-24b vs qwen3-8b | +0.0137 | 0.0045 | +3.05 | 0.007 | [+0.0049, +0.0226] | ** |
| mistral-small-24b vs ministral-14b | -0.0246 | 0.0045 | -5.45 | < .001 | [-0.0334, -0.0157] | ** |

### Answer Relevancy

Random effects variance (σ²_q): 0.002154
N observations: 720

| Parameter | Coef | SE | z | p | 95% CI | Sig |
|-----------|------|----|---|---|--------|-----|
| llm=ministral-14b | +0.0025 | 0.0082 | +0.31 | 0.756 | [-0.0135, +0.0186] |  |
| llm=mistral-small-24b | -0.0088 | 0.0082 | -1.07 | 0.284 | [-0.0248, +0.0073] |  |
| chunk=1000 | -0.0003 | 0.0067 | -0.05 | 0.960 | [-0.0134, +0.0128] |  |
| k=4 | +0.0064 | 0.0067 | +0.96 | 0.337 | [-0.0067, +0.0195] |  |
| weights=70/30 | +0.0092 | 0.0067 | +1.38 | 0.167 | [-0.0039, +0.0223] |  |

**Pairwise LLM contrasts (answer_relevancy, Bonferroni corrected)**

| Comparison | Diff | SE | z | p (adj) | 95% CI | Sig |
|-----------|------|----|---|---------|--------|-----|
| ministral-14b vs qwen3-8b | +0.0025 | 0.0082 | +0.31 | 1.000 | [-0.0135, +0.0186] |  |
| mistral-small-24b vs qwen3-8b | -0.0088 | 0.0082 | -1.07 | 0.852 | [-0.0248, +0.0073] |  |
| mistral-small-24b vs ministral-14b | -0.0113 | 0.0082 | -1.38 | 0.501 | [-0.0274, +0.0047] |  |

### Faithfulness

Random effects variance (σ²_q): 0.002901
N observations: 720

| Parameter | Coef | SE | z | p | 95% CI | Sig |
|-----------|------|----|---|---|--------|-----|
| llm=ministral-14b | +0.0021 | 0.0078 | +0.27 | 0.789 | [-0.0132, +0.0174] |  |
| llm=mistral-small-24b | +0.0125 | 0.0078 | +1.60 | 0.110 | [-0.0028, +0.0278] |  |
| chunk=1000 | -0.0069 | 0.0064 | -1.08 | 0.280 | [-0.0194, +0.0056] |  |
| k=4 | -0.0202 | 0.0064 | -3.16 | 0.002 | [-0.0327, -0.0077] | ** |
| weights=70/30 | +0.0032 | 0.0064 | +0.51 | 0.613 | [-0.0093, +0.0157] |  |

### Contextual Recall

Random effects variance (σ²_q): 0.019174
N observations: 716

| Parameter | Coef | SE | z | p | 95% CI | Sig |
|-----------|------|----|---|---|--------|-----|
| llm=ministral-14b | -0.0097 | 0.0168 | -0.58 | 0.563 | [-0.0426, +0.0232] |  |
| llm=mistral-small-24b | -0.0111 | 0.0168 | -0.66 | 0.506 | [-0.0440, +0.0217] |  |
| chunk=1000 | +0.0944 | 0.0137 | +6.89 | < .001 | [+0.0676, +0.1212] | ** |
| k=4 | +0.1193 | 0.0137 | +8.71 | < .001 | [+0.0924, +0.1461] | ** |
| weights=70/30 | -0.0049 | 0.0137 | -0.36 | 0.722 | [-0.0317, +0.0220] |  |

---
## Group A — LLM Answer Quality (A)
*Composite of: actionability, correctness, answer_relevancy (equal weights, -1 excluded)*

### LLM Answer Quality (A)

Random effects variance (σ²_q): 0.000941
N observations: 720

| Parameter | Coef | SE | z | p | 95% CI | Sig |
|-----------|------|----|---|---|--------|-----|
| llm=ministral-14b | +0.0299 | 0.0037 | +8.07 | < .001 | [+0.0226, +0.0371] | ** |
| llm=mistral-small-24b | -0.0020 | 0.0037 | -0.55 | 0.585 | [-0.0093, +0.0052] |  |
| chunk=1000 | +0.0060 | 0.0030 | +2.00 | 0.045 | [+0.0001, +0.0120] | * |
| k=4 | +0.0095 | 0.0030 | +3.14 | 0.002 | [+0.0036, +0.0154] | ** |
| weights=70/30 | +0.0054 | 0.0030 | +1.80 | 0.072 | [-0.0005, +0.0114] |  |

**Pairwise LLM contrasts (LLM Quality Composite, Bonferroni corrected)**

| Comparison | Diff | SE | z | p (adj) | 95% CI | Sig |
|-----------|------|----|---|---------|--------|-----|
| ministral-14b vs qwen3-8b | +0.0299 | 0.0037 | +8.07 | < .001 | [+0.0226, +0.0371] | ** |
| mistral-small-24b vs qwen3-8b | -0.0020 | 0.0037 | -0.55 | 1.000 | [-0.0093, +0.0052] |  |
| mistral-small-24b vs ministral-14b | -0.0319 | 0.0037 | -8.62 | < .001 | [-0.0392, -0.0246] | ** |

---
## Group B — RAG Retrieval Quality (B)
*Composite of: contextual_recall, faithfulness (equal weights, -1 excluded)*

### RAG Retrieval Quality (B)

Random effects variance (σ²_q): 0.004218
N observations: 720

| Parameter | Coef | SE | z | p | 95% CI | Sig |
|-----------|------|----|---|---|--------|-----|
| llm=ministral-14b | -0.0044 | 0.0093 | -0.47 | 0.637 | [-0.0225, +0.0138] |  |
| llm=mistral-small-24b | +0.0001 | 0.0093 | +0.01 | 0.992 | [-0.0181, +0.0182] |  |
| chunk=1000 | +0.0443 | 0.0076 | +5.86 | < .001 | [+0.0295, +0.0592] | ** |
| k=4 | +0.0498 | 0.0076 | +6.58 | < .001 | [+0.0350, +0.0646] | ** |
| weights=70/30 | -0.0005 | 0.0076 | -0.06 | 0.950 | [-0.0153, +0.0143] |  |

---
## Overall Weighted Score
*0.25×actionability + 0.25×correctness + 0.2×answer_relevancy + 0.15×faithfulness + 0.15×contextual_recall (renormalised if metric unavailable)*

### Overall Weighted Score

Random effects variance (σ²_q): 0.000896
N observations: 720

| Parameter | Coef | SE | z | p | 95% CI | Sig |
|-----------|------|----|---|---|--------|-----|
| llm=ministral-14b | +0.0210 | 0.0037 | +5.70 | < .001 | [+0.0138, +0.0283] | ** |
| llm=mistral-small-24b | -0.0012 | 0.0037 | -0.33 | 0.745 | [-0.0084, +0.0060] |  |
| chunk=1000 | +0.0176 | 0.0030 | +5.84 | < .001 | [+0.0117, +0.0235] | ** |
| k=4 | +0.0220 | 0.0030 | +7.29 | < .001 | [+0.0161, +0.0279] | ** |
| weights=70/30 | +0.0037 | 0.0030 | +1.22 | 0.224 | [-0.0022, +0.0096] |  |

**Pairwise LLM contrasts (Overall, Bonferroni corrected)**

| Comparison | Diff | SE | z | p (adj) | 95% CI | Sig |
|-----------|------|----|---|---------|--------|-----|
| ministral-14b vs qwen3-8b | +0.0210 | 0.0037 | +5.70 | < .001 | [+0.0138, +0.0283] | ** |
| mistral-small-24b vs qwen3-8b | -0.0012 | 0.0037 | -0.33 | 1.000 | [-0.0084, +0.0060] |  |
| mistral-small-24b vs ministral-14b | -0.0222 | 0.0037 | -6.02 | < .001 | [-0.0295, -0.0150] | ** |

---
## Bad Question Detection

### Systematically Hard Questions (BLUPs)

Random intercept stats: mean=0.000, SD=0.028, cutoff (<mean−1.5×SD) = -0.043

| Q# | BLUP | Question |
|----|------|---------|
| 13 | -0.061 | Why are my chickens losing feathers? |
| 2 | -0.044 | How many eggs do chickens lay per day? |

