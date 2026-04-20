# Phase 2b — Prompt Variant Validation (frankenstein vs originals)

Model: `score ~ C(variant) + (1|q_num)`
Reference level: `frankenstein` (current production prompt)
Judge: `openai/gpt-4o-mini-2024-07-18` via OpenRouter
Test set: 45 corpus-grounded synthetic goldens (goldens_synth_v1.json)
RAG: on (hybrid 70/30, k=4). Sensors: off.
Family-wise correction: Holm across all non-reference contrasts (2 metrics x 3 variants = 6 tests).

## Descriptive Means

| Variant | N | Actionability | Correctness |
|---------|---|---------------|-------------|
| frankenstein | 45 | 0.9111 | 0.6800 |
| concise | 45 | 0.9178 | 0.6600 |
| expert | 45 | 0.8978 | 0.7067 |
| structured | 45 | 0.9000 | 0.6289 |

## LME Fixed Effects

### Actionability

| Parameter | Coef (delta) | SE | z | p (Wald) | 95% CI |
|-----------|-------------|----|----|----------|--------|
| Intercept | +0.9111 | 0.0084 | +108.35 | < .001 | [+0.8946, +0.9276] |
| variant=concise | +0.0067 | 0.0103 | +0.64 | 0.519 | [-0.0136, +0.0269] |
| variant=expert | -0.0133 | 0.0103 | -1.29 | 0.198 | [-0.0336, +0.0069] |
| variant=structured | -0.0111 | 0.0103 | -1.07 | 0.283 | [-0.0314, +0.0092] |

### Correctness

| Parameter | Coef (delta) | SE | z | p (Wald) | 95% CI |
|-----------|-------------|----|----|----------|--------|
| Intercept | +0.6800 | 0.0135 | +50.42 | < .001 | [+0.6536, +0.7064] |
| variant=concise | -0.0200 | 0.0122 | -1.63 | 0.102 | [-0.0440, +0.0040] |
| variant=expert | +0.0267 | 0.0122 | +2.18 | 0.029 | [+0.0027, +0.0506] |
| variant=structured | -0.0511 | 0.0122 | -4.18 | < .001 | [-0.0751, -0.0271] |

## Holm-Corrected Contrasts (vs frankenstein)

| Contrast | Metric | delta | Wald p (raw) | Holm-adjusted p |
|----------|--------|-------|--------------|-----------------|
| concise | actionability | +0.0067 | 0.519 | 0.593 |
| expert | actionability | -0.0133 | 0.198 | 0.593 |
| structured | actionability | -0.0111 | 0.283 | 0.593 |
| concise | correctness | -0.0200 | 0.102 | 0.409 |
| expert | correctness | +0.0267 | 0.029 | 0.147 |
| structured | correctness | -0.0511 | < .001 | < .001 |

