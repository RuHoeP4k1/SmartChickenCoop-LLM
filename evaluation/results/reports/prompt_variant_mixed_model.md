# Mixed Effects Analysis — Prompt Variants

Model: `score ~ C(variant) + (1|q_num)`
Random intercept per question removes between-question noise from variant effect estimates.
Reference level: variant=baseline

Significance: \* p<0.05  \*\* p<0.01  (pairwise contrasts: Bonferroni corrected)

## Descriptive Summary

| Variant | N | Actionability (mean) | Correctness (mean) | Combined (mean) |
|---------|---|----------------------|--------------------|-----------------|
| baseline | 30 | 0.8267 | 0.6567 | 0.7417 |
| concise | 30 | 0.8633 | 0.6933 | 0.7783 |
| expert | 30 | 0.8667 | 0.7567 | 0.8117 |
| structured | 30 | 0.8900 | 0.7433 | 0.8167 |

## Individual Metrics

### Actionability

Random effects variance (σ²_q): 0.001243
N observations: 120

| Parameter | Coef | SE | z | p | 95% CI | Sig |
|-----------|------|----|---|---|--------|-----|
| variant=concise | +0.0367 | 0.0159 | +2.31 | 0.021 | [+0.0056, +0.0678] | * |
| variant=expert | +0.0400 | 0.0159 | +2.52 | 0.012 | [+0.0089, +0.0711] | * |
| variant=structured | +0.0633 | 0.0159 | +3.99 | < .001 | [+0.0322, +0.0944] | ** |

**Pairwise variant contrasts (Actionability, Bonferroni corrected)**

| Comparison | Diff | SE | z | p (adj) | 95% CI | Sig |
|-----------|------|----|---|---------|--------|-----|
| concise vs baseline | +0.0367 | 0.0159 | +2.31 | 0.125 | [+0.0056, +0.0678] |  |
| expert vs baseline | +0.0400 | 0.0159 | +2.52 | 0.070 | [+0.0089, +0.0711] |  |
| structured vs baseline | +0.0633 | 0.0159 | +3.99 | < .001 | [+0.0322, +0.0944] | ** |
| expert vs concise | +0.0033 | 0.0159 | +0.21 | 1.000 | [-0.0278, +0.0344] |  |
| structured vs concise | +0.0267 | 0.0159 | +1.68 | 0.556 | [-0.0044, +0.0578] |  |
| structured vs expert | +0.0233 | 0.0159 | +1.47 | 0.848 | [-0.0078, +0.0544] |  |

### Correctness

Random effects variance (σ²_q): 0.015216
N observations: 120

| Parameter | Coef | SE | z | p | 95% CI | Sig |
|-----------|------|----|---|---|--------|-----|
| variant=concise | +0.0367 | 0.0295 | +1.24 | 0.214 | [-0.0212, +0.0945] |  |
| variant=expert | +0.1000 | 0.0295 | +3.39 | < .001 | [+0.0422, +0.1578] | ** |
| variant=structured | +0.0867 | 0.0295 | +2.94 | 0.003 | [+0.0288, +0.1445] | ** |

**Pairwise variant contrasts (Correctness, Bonferroni corrected)**

| Comparison | Diff | SE | z | p (adj) | 95% CI | Sig |
|-----------|------|----|---|---------|--------|-----|
| concise vs baseline | +0.0367 | 0.0295 | +1.24 | 1.000 | [-0.0212, +0.0945] |  |
| expert vs baseline | +0.1000 | 0.0295 | +3.39 | 0.004 | [+0.0422, +0.1578] | ** |
| structured vs baseline | +0.0867 | 0.0295 | +2.94 | 0.020 | [+0.0288, +0.1445] | * |
| expert vs concise | +0.0633 | 0.0295 | +2.15 | 0.191 | [+0.0055, +0.1212] |  |
| structured vs concise | +0.0500 | 0.0295 | +1.69 | 0.541 | [-0.0078, +0.1078] |  |
| structured vs expert | -0.0133 | 0.0295 | -0.45 | 1.000 | [-0.0712, +0.0445] |  |

### Combined

Random effects variance (σ²_q): 0.004795
N observations: 120

| Parameter | Coef | SE | z | p | 95% CI | Sig |
|-----------|------|----|---|---|--------|-----|
| variant=concise | +0.0367 | 0.0170 | +2.16 | 0.031 | [+0.0034, +0.0700] | * |
| variant=expert | +0.0700 | 0.0170 | +4.12 | < .001 | [+0.0367, +0.1033] | ** |
| variant=structured | +0.0750 | 0.0170 | +4.41 | < .001 | [+0.0417, +0.1083] | ** |

**Pairwise variant contrasts (Combined, Bonferroni corrected)**

| Comparison | Diff | SE | z | p (adj) | 95% CI | Sig |
|-----------|------|----|---|---------|--------|-----|
| concise vs baseline | +0.0367 | 0.0170 | +2.16 | 0.186 | [+0.0034, +0.0700] |  |
| expert vs baseline | +0.0700 | 0.0170 | +4.12 | < .001 | [+0.0367, +0.1033] | ** |
| structured vs baseline | +0.0750 | 0.0170 | +4.41 | < .001 | [+0.0417, +0.1083] | ** |
| expert vs concise | +0.0333 | 0.0170 | +1.96 | 0.299 | [+0.0000, +0.0666] |  |
| structured vs concise | +0.0383 | 0.0170 | +2.26 | 0.145 | [+0.0050, +0.0716] |  |
| structured vs expert | +0.0050 | 0.0170 | +0.29 | 1.000 | [-0.0283, +0.0383] |  |

---
## Bad Question Detection

### Systematically Hard Questions (BLUPs)

Random intercept stats: mean=-0.000, SD=0.061, cutoff (<mean−1.5×SD) = -0.092

| Q# | BLUP | Question |
|----|------|---------|
| 8 | -0.122 | Why did my chicken stop eating? |

