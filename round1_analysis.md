# Round 1 Sweep Analysis

**Timestamp:** 20260314_092600  
**Judge:** meta-llama/llama-3.3-70b-instruct  
**Questions per config:** 30  
**Configs completed:** 24  

---

## Winner Config

| Factor | Value |
|--------|-------|
| LLM | `openrouter/mistralai/ministral-14b-2512` |
| Chunk size | 1000 |
| k | 2 |
| Weights | 70/30 |

**Combined score:** 0.9667  
**Actionability:** 0.9900  
**Correctness:** 0.9433

---

## A. Main Effects

```

======================================================================
A. MAIN EFFECTS  (metric: combined)
======================================================================
Factor          Level            Avg Score      N     Range
----------------------------------------------------------------------
llm_model       ministral-14b       0.9592      8   +0.0435
                mistral-small-24b      0.9169      8          
                qwen3-8b            0.9156      8          

k               4                   0.9361     12   +0.0111
                2                   0.9250     12          

chunk_size      1000                0.9352     12   +0.0092
                600                 0.9260     12          

weights         70/30               0.9324     12   +0.0036
                pure-sem            0.9288     12          

```

---

## B. ANOVA

```

============================================================
B. ONE-WAY ANOVA  (H₀: all levels produce the same score)
============================================================
Factor            F-stat    p-value   Significant
------------------------------------------------------------
llm_model          34.85     0.0000  *** SIGNIFICANT
k                   1.34     0.2592  
chunk_size          0.91     0.3505  
weights             0.13     0.7185  
============================================================
Threshold: p < 0.05
```

---

## C. Two-Way Interactions

```

C. TWO-WAY INTERACTIONS
  (fewer than 2 significant factors — skipping)

```

---

## D. Top 10 Configurations

```

================================================================================
D. TOP 10 CONFIGURATIONS  (by avg combined score)
================================================================================
Rank  LLM             Chunk  k   Weights      Action  Correct  Combined
--------------------------------------------------------------------------------
1     openrouter/mis  1000   2   70/30        0.9900   0.9433    0.9667
2     openrouter/mis  600    4   70/30        0.9933   0.9333    0.9633
3     openrouter/mis  600    2   70/30        0.9900   0.9333    0.9617
4     openrouter/mis  1000   2   pure-sem     0.9833   0.9400    0.9617
5     openrouter/mis  1000   4   pure-sem     0.9900   0.9300    0.9600
6     openrouter/mis  600    2   pure-sem     0.9800   0.9300    0.9550
7     openrouter/mis  600    4   pure-sem     0.9867   0.9233    0.9550
8     openrouter/mis  1000   4   70/30        0.9633   0.9367    0.9500
9     openrouter/qwe  1000   4   pure-sem     0.9600   0.9167    0.9383
10    openrouter/qwe  1000   4   70/30        0.9700   0.9033    0.9367
================================================================================
```

---

## Prompt Review Checklist

- [ ] Read answers for top 3 configs — what does a good answer look like?
- [ ] Read answers for bottom 3 configs — what failure modes appear?
- [ ] Do small models (0.5b) fail on retrieval or generation?
- [ ] Are larger models over-verbose? Check length calibration.
- [ ] Any prompt phrasing that systematically confuses models?
- [ ] Update `prompts.py` if needed. Tag commit: `prompt-v2`.

---

## Round 2 Plan

1. Fix low-impact factors (see ANOVA) at their best level.
2. Run full factorial on top 2–3 significant factors.
3. Re-run winner config with new prompt to measure prompt impact.
4. Merge human ranking CSV if available.