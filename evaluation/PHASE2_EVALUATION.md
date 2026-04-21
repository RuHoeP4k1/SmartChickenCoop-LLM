# Phase 2 Evaluation Plan

<!-- AUTO-GENERATED -->
## Status

| Axis | Status | Output file |
|------|--------|-------------|
| Axis 1 — Prompt variants (G-Eval scoring) | Complete | `results/prompt_variant_results.json` |
| Axis 1 — Human pairwise ranking + mixed model analysis | Complete | `results/prompt_variant_mixed_model.md` |
| Axis 2 — Sensor awareness end-to-end (19 scenarios, incl. routing scores) | Complete | `results/sensor_awareness_results.json` |
| Axis 2 — Keyword vs LLM routing comparison (3 scenarios) | Complete | `results/sensor_routing_comparison.json` |

### Axis 2 — Sensor Routing: Keyword vs LLM Classifier

The central Phase 2 question for sensor awareness: **should sensor context injection be decided by a rule-based keyword filter or by an LLM call?**

Two implementations exist in `backend/sensor_filter.py`:

| Approach | Function | Latency | Cost per call | Implementation |
|----------|----------|---------|---------------|----------------|
| Keyword filter | `should_include_sensors()` | ~0 ms | $0 | 80+ keyword rules across 7 sets, priority-ordered |
| LLM classifier | `llm_route_sensors()` | ~400–1100 ms | ~$0.00009 | ministral-14b via OpenRouter, INCLUDE/EXCLUDE prompt |

The LLM classifier uses the same model as production with a compact single-turn prompt that receives a one-line sensor summary and the user query. It falls back to keyword routing on parse error.

**Routing results across all 19 sensor awareness scenarios** (`sensor_awareness_results.json`, `routing_correct` field):

**Keyword filter accuracy: 18/19 (94.7%)**. Single failure: S13 (H₂S critical, encyclopedic question about ammonia — keyword rule 4 overrides the general knowledge suppression, incorrectly includes sensor context).

All 19 scenarios were run through both approaches (`compare_sensor_routing.py`, `sensor_routing_comparison.json`, model: `ministral-14b-2512`):

| Approach | Correct | Accuracy | Failure |
|----------|---------|----------|---------|
| Keyword filter | 18/19 | 94.7% | S13 |
| LLM classifier | 19/19 | 100% | — |

LLM latency ranged from 171–1250 ms (median ~194 ms, one outlier at S14: 1250 ms). Cost: ~$0.00009 per call. The keyword filter is the production default; the LLM classifier is configurable via `SENSOR_ROUTER_MODEL` env var.

**Reflection — keyword vs LLM routing:**

The cost argument for LLM routing is strong. At ~$0.00009 per call, routing 1,000 queries costs under $0.10 — negligible compared to the main LLM inference cost and well within any realistic usage budget. The latency overhead (~400–1100 ms) is more relevant than cost for a conversational interface, but remains acceptable for an advisory system that is not real-time.

The more fundamental advantage of LLM routing is flexibility. The keyword filter requires explicit enumeration of every phrasing pattern that should trigger sensor inclusion — this works well for common cases but becomes brittle at the edges. Phrasing variations, new sensor types, or languages not covered by the keyword sets all require manual maintenance of the rule lists. The LLM classifier generalises from the routing rules described in the prompt rather than from an exhaustive list, making it more robust to novel inputs without code changes. This is particularly relevant as the sensor schema expands (e.g. egg count, door state, H₂S were all added after the initial keyword sets were written).

The single keyword filter failure (S13: H₂S critical + encyclopedic question about ammonia) illustrates this rigidity directly: rule 4 ("critical condition + chicken topic → INCLUDE") was written before H₂S was added to the schema, and its broad scope incorrectly overrides the general knowledge suppression for "what causes ammonia buildup?" — an encyclopedic question that should not receive sensor context. The LLM classifier correctly returns EXCLUDE for this case, reasoning about the question's encyclopedic nature against the sensor state without needing an explicit rule. This is the concrete advantage: the LLM classifier achieves 19/19 (100%) where the keyword filter reaches 18/19 (94.7%), and the one case it fixes is precisely the kind of edge case that keyword enumeration cannot anticipate without unbounded rule growth.

### Sensor Awareness End-to-End Results

Full pipeline evaluation: 19 scenarios across all sensor types. **16/19 pass (84.2%).**

| Scenario | Result | Failure reason |
|----------|--------|----------------|
| S01–S02: Normal sensors, general question | pass | — |
| S03: Normal sensors, "is my coop ok?" | **fail** | Model over-hedges: says "can't confirm" instead of confirming normal reading |
| S04–S10: Warning/critical sensor states | pass | Sensor values cited correctly, urgency appropriate |
| S11: General encyclopedic question, normal sensors | **fail** | Urgency miscalibration: encyclopedic answer tone scored below threshold |
| S12: H₂S critical (52 ppm), user reports smell | pass | Cites ppm value, urgent language |
| S13: H₂S critical, encyclopedic ammonia question | **fail** | Routing_correct false: critical H₂S overrides encyclopedic exclusion rule |
| S14–S19: Mold/door/egg/resource/status scenarios | pass | — |

Three systematic failure modes: (1) over-hedging when all readings are normal; (2) urgency tone miscalibration on informational questions; (3) H₂S critical state triggering sensor inclusion for encyclopedic queries (rule 4 conflict with general knowledge suppression).
<!-- END AUTO-GENERATED -->

---

Once the configuration sweep (model / chunk_size / k / weights) is complete, two independent evaluation axes remain.

---

## Axis 1 — Prompt design

**Goal:** find which prompt template produces the best answers, holding the RAG configuration constant.

Four variants of the `SIMPLE_PROMPT` are tested (the most-used path — no live sensor context):

| Variant | Design hypothesis |
|---|---|
| `baseline` | Current production prompt (reference) |
| `structured` | Does explicit output structure (short answer / steps / conditional vet) improve actionability? |
| `concise` | Is baseline's explicit guidance necessary, or does minimal framing work equally well? |
| `expert` | Does experienced domain-expert positioning (researcher + hands-on keeper) improve accuracy? |

Evaluation is done in two ways:

1. **DeepEval G-Eval scoring** — Actionability + Correctness (AI judge via Kimi 2.6). These measure LLM answer quality independent of retrieval.
2. **Human pairwise preference** — blinded A vs B comparisons where raters don't know which variant they're seeing. Results are aggregated into win rates and ELO scores per variant.

### Human ranking methodology

Raters evaluate **answer quality**, not factual correctness. The rating criteria are:
- Clarity — is the answer easy to read and understand?
- Structure — is it well-organised? Does formatting help or hurt?
- Usefulness — would it help a chicken keeper take action?
- Tone — appropriate level of urgency and detail?

Raters are explicitly told they are **not** judging factual accuracy, but can leave a free-text comment on either answer if something concerns them.

**Implementation:** a standalone web app (`ranking_app/`) built with FastAPI + React, shared via ngrok. Raters access it by name — their progress is saved to Supabase so they can close and resume across sessions. A/B assignment is randomised per `(rater_id, pair_id)` and resolved server-side so variant labels never reach the client.

**ELO scoring:** all variants start at 1000. Each pairwise vote is treated as a match:

```
E_a = 1 / (1 + 10^((elo_b - elo_a) / 400))   # expected score for A
elo_a += K * (actual - expected)               # K = 16
```

- A win → actual = 1, tie → 0.5, loss → 0
- K = 16 (conservative factor for stable ratings) — appropriate for small N with multiple raters
- A variant only climbs by beating strong opponents; beating weaker ones yields smaller gains
- Starting ELO of 1000 is arbitrary; only relative differences matter

Win rate is reported alongside ELO as a simpler cross-check. Both metrics should agree on ranking order; large divergence between them indicates uneven matchup distribution.

To retrieve results: `python ranking_app/analyze.py` or `GET /results` on the ranking app backend.

### Results — Human Ranking + Mixed Effects Analysis

**Human preference ranking** (37 raters, 644 total votes across 178 rated pairs):

| Variant | ELO Rating | Wins | Losses | Ties | Win Rate |
|---------|-----------|------|--------|------|----------|
| concise | 1033.4 | 138 | 150 | 25 | 44.1% |
| structured | 1011.5 | 179 | 126 | 14 | 56.1% |
| expert | 967.5 | 173 | 149 | 13 | 51.6% |
| baseline | 987.6 | 117 | 182 | 22 | 36.4% |

Human raters ranked `structured` highest (56.1% win rate), followed by `expert` (51.6%), with `baseline` weakest (36.4%). `concise` had highest ELO but lower win rate, indicating mixed strength across pairings.

**G-Eval + human combined via linear mixed model** (random intercept per question, 30 questions × 4 variants = 120 observations). Formula: `score ~ C(variant) + (1|q_num)`.

| Variant | Actionability | Correctness | Combined |
|---------|---------------|------------|----------|
| baseline | 0.827 | 0.657 | 0.742 |
| concise | 0.863 | 0.693 | 0.778 |
| expert | 0.867 | 0.757 | 0.812 |
| structured | 0.890 | 0.743 | 0.817 |

**Key findings:**

1. **Human vs automated consensus:** Humans and G-Eval largely agree. `structured` wins on actionability (0.890 vs 0.827 baseline, p<0.001) and is human-ranked #1 by win rate (56.1%).
2. **Actionability:** `structured` (+0.0633, p<0.001) significantly outperforms baseline. All variants improve on baseline; `structured` is best.
3. **Correctness:** `expert` (+0.1000, p<0.001) and `structured` (+0.0867, p=0.003) both significantly outperform baseline. After Bonferroni correction, expert remains significant (p=0.004).
4. **Combined score:** `expert` (+0.0700) and `structured` (+0.0750) both significantly better than baseline (p<0.001), even after correction. No significant difference between them (p=1.000).
5. **Bad question detected:** Q8 ("Why did my chicken stop eating?") consistently scored lower (BLUP = −0.122, >1.5 SD below mean). Inherently difficult; may require specialized knowledge not well-covered in knowledge base.

**Recommendation:** Adopt `structured` prompt. It achieves both the highest win rate from human raters (56.1%) and best actionability from G-Eval (0.890), while being simpler to maintain than `expert`. Statistical significance survives Bonferroni correction for multiple comparisons.

### Qualitative insights from human raters

**Why structured won:** Raters valued the explicit "short answer → steps → vet decision" layout. Markdown formatting (bullets, bold) improves scannability and helps users act immediately without reading the entire response.

**Key limitation identified:** Generated answers were **too comprehensive**. When the LLM has broad knowledge (e.g., "what bedding should I use?"), it tends to list *all* options in exhaustive detail. Raters feedback: anticipate follow-up questions on bullet points rather than preemptively covering everything. Better pattern: **identify core issues → summarize top 2–3 actions → prompt user to ask for specifics** (e.g., "which of these is best for my wet climate?").

**Critical evaluation gap:** These 30 questions were evaluated **without live sensor data**. The real production pipeline injects sensor readings when appropriate (temperature, humidity, resource levels, etc.). This evaluation measured RAG + prompt quality in isolation. **Full pipeline evaluation** (sensor routing + sensor-aware answer generation) would require:
- Controlled sensor scenarios (we have 19 in Axis 2)
- Eval criteria for sensor value citation and urgency calibration
- Much broader permutation space (normal sensors × 4 prompt variants × 3 sensor scenarios = 12+; critical scenarios × variants = more)

This explains why Axis 2 (sensor awareness) is a separate evaluation axis — the interaction between live data and prompt phrasing is complex enough to warrant its own test suite.

**Next steps:** Design a new hybrid prompt combining all four variants' strengths. ✓ COMPLETED (2026-03-27)

### Implementation: Hybrid Prompts + Semantic Routing

**Prompts updated (2026-03-27):**
- `SIMPLE_PROMPT` — structured layout, concise, anticipates follow-ups
- `MAIN_PROMPT` — uses sensor data to validate/prioritise knowledge base claims
- `EMERGENCY_PROMPT` — critical reading leads; knowledge base supports actions only

**Sensor routing switched to semantic (LLM-based) classification** (2026-03-27):
- Default routing mode: `SENSOR_ROUTING_MODE=llm` (already set in `rag_functions.py:549`)
- Uses `llm_route_sensors()` — LLM classifier achieves 100% accuracy (19/19 scenarios) vs keyword filter's 94.7% (18/19)
- Single failure in keyword approach (S13: H₂S critical + encyclopedic ammonia question) is now fixed by LLM classifier
- Fallback: on LLM error, reverts to keyword routing (`should_include_sensors`)
- Model configurable via `SENSOR_ROUTER_MODEL` env var (defaults to main LLM model)

### Production prompt anatomy — what the hybrid actually inherits

The resulting `STANDARD_PROMPT` in `backend/prompts.py` is a post-hoc synthesis of the three winning variants rather than a clean copy of any one. The table below maps each design decision back to its source:

| Element | Production `STANDARD_PROMPT` | Source variant |
|---|---|---|
| Output skeleton | `**The short answer:** (1–2 sentences)` + `**Steps to take:**` | **structured** |
| Informational escape hatch | "write a concise explanation instead of forcing action steps" | **structured** |
| Vet restraint rule | "Only mention a vet for genuine emergencies — visible injury, suspected contagious disease, or a bird in acute distress" | **expert** (calibrated restraint, not a generic disclaimer) |
| Persona framing | "friendly, knowledgeable assistant for hobby chicken keepers" | **baseline** (short form) |
| Instruction density | Moderate — explicit safety rules but no deep domain-authority backstory | **concise** trimming applied to **baseline** foundation |
| Sensor-specific guards | "never present reference knowledge as something you are currently observing"; `_SENSOR_ANCHOR` / `_SENSOR_PRIORITISE` dynamic injection | Production-only — not tested in any variant |

**What was dropped in the merge:** The `expert` variant's domain-authority persona framing ("part researcher, part seasoned flock keeper… you know the difference between textbook advice and what actually works at 6 a.m.") was not carried over. This framing may have been responsible for expert's correctness advantage (0.757 vs structured's 0.743 G-Eval score). The current prompt gets the structural win from `structured` and the vet calibration from `expert`, but loses the confidence signal that the expert persona embedded in the model's output register.

**Gap not closed by Phase 2 eval:** The sensor-specific guards (`_SENSOR_ANCHOR`, `_SENSOR_PRIORITISE`, the "never present reference knowledge as an observation" rule) were added after the variant tests and were never scored. Their effect on answer quality — particularly hallucination suppression when sensor context is injected — remains unmeasured.

---

## Axis 2 — Sensor / real-time awareness

**Goal:** verify that the system correctly integrates live sensor data, independent of RAG retrieval quality.

Even with a perfect knowledge base, the system needs to:

1. **Include** sensor context when the reading is relevant to the question
2. **Exclude** sensor context for general knowledge questions with normal readings
3. **Cite** the actual sensor values (e.g. "35.2°C" should appear in the answer)
4. **Match urgency to severity** — critical readings → urgent language; normal readings → calm tone
5. **Not fabricate** sensor values when no sensor context was injected into the prompt

Nineteen test scenarios cover these cases using mocked sensor snapshots (no Pi or database needed):

| Scenario | Sensor state | Question type | Key check |
|---|---|---|---|
| S01–S02 | All normal | General / breed | Sensors correctly excluded |
| S03 | All normal | "Is my coop ok?" | Sensors included, says "all normal" |
| S04 | Temp warning (27.4°C) | "Is my coop warm?" | Cites 27°C, mild concern tone |
| S05 | Temp critical (35.2°C) | Chickens panting | Cites 35°C, urgent language |
| S06 | Temp critical | Why not laying? | Critical overrides → sensors still included |
| S07 | Humidity critical (91%) | "Are my chickens ok?" | Cites 91%, urgent language |
| S08 | Feeder empty | "Is my feeder ok?" | Mentions empty feeder |
| S09 | Waterer low | Feed question | Sensors correctly excluded (not mentioned in query) |
| S10 | All critical | "How are my chickens?" | All alerts cited, maximum urgency |
| S11 | All normal | General temp question | Sensor value (21.5°C) not fabricated in answer |
| S12 | H₂S critical (52 ppm) | Ammonia smell reported | Cites 52 ppm, urgent language |
| S13 | H₂S critical | Encyclopedic ammonia question | Sensors excluded (encyclopedic, no live context) |
| S14 | Mold risk critical | "Is my coop okay?" | Include mold risk, mild concern tone |
| S15 | Door open | "Did I close the door?" | Include, report door is open |
| S16 | 3 eggs detected | "Any eggs today?" | Include, cite egg count |
| S17 | Waterer low (18%) | "Is my waterer okay?" | Include, mention low level |
| S18 | All critical | Non-chicken topic ("What breed?") | Sensors excluded (off-topic) |
| S19 | All normal | Full status request | Include, confirm all normal |

---

## Files

| File | Purpose |
|---|---|
| `phase2_prompts/evaluate_variants.py` | DeepEval G-Eval scoring of 4 prompt variants + pairwise export |
| `phase2_prompts/evaluate_sensor_awareness.py` | Sensor awareness scoring across 19 scenarios |
| `phase2_prompts/human_ranking.py` | CLI pairwise ranking tool (ELO + win rates) |
| `phase2_prompts/compare_sensor_routing.py` | Keyword vs LLM routing comparison (19 scenarios × 3 runs) |
| `phase2_prompts/mixed_model.py` | Mixed effects analysis of prompt variant scores |

---

## Workflow

### Step 1 — Run prompt variant evaluation

```bash
python evaluation/phase2_prompts/evaluate_variants.py --export-pairs
```

This generates two files:
- `results/prompt_variant_results.json` — automated scores per variant per question
- `results/prompt_pairs.json` — blinded A/B pairs for human ranking

### Step 2 — Human pairwise ranking

Ratings are collected via the standalone web app (`ranking_app/`). Start it with:

```bash
# Terminal 1 — backend
cd ranking_app && uvicorn main:app --reload --port 8001

# Terminal 2 — frontend
cd ranking_app/frontend && npm run dev -- --host

# Terminal 3 — share via ngrok
ngrok http --domain=unconsecrative-prorevision-lonny.ngrok-free.dev 5174
```

Raters access the URL, enter their name, and rate pairs in a blinded interface. Progress is saved to Supabase — they can stop and resume at any time. A leaderboard is visible in-app.

View aggregated results at any time:

```bash
python ranking_app/analyze.py
```

### Step 3 — Sensor awareness evaluation

```bash
python evaluation/phase2_prompts/evaluate_sensor_awareness.py
```

Results are saved to `results/sensor_awareness_results.json`.

Use `--verbose` to see the full answer text for each scenario:

```bash
python evaluation/phase2_prompts/evaluate_sensor_awareness.py --verbose
```

### Step 4 — Smoke tests (optional, before full runs)

```bash
# Test only 5 questions for prompt variants
python evaluation/phase2_prompts/evaluate_variants.py --n-questions 5

# Test only 4 sensor scenarios
python evaluation/phase2_prompts/evaluate_sensor_awareness.py --n-scenarios 4
```

---

## Dependencies

All scripts inherit the same environment as the rest of the evaluation pipeline. No additional packages needed beyond what is already in `requirements.txt`.

The sensor awareness evaluation uses mocked sensor data — no database connection or Raspberry Pi hardware is required.
