# Phase 2 Evaluation Plan

<!-- AUTO-GENERATED -->
## Status

| Axis | Status | Output file |
|------|--------|-------------|
| Axis 1 — Prompt variants (G-Eval scoring) | Complete | `results/prompt_variant_results.json` |
| Axis 1 — Human pairwise ranking | Pending | `results/prompt_pairs.json` |
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

1. **DeepEval G-Eval scoring** — Actionability + Correctness (AI judge via Claude Haiku). These measure LLM answer quality independent of retrieval.
2. **Human pairwise preference** — blinded A vs B comparisons where raters don't know which variant they're seeing. Results are aggregated into win rates and ELO scores. This human evaluation method can be extended beyond only evaluating the different prompt variants ofc!

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
| `evaluate_prompt_variants.py` | DeepEval G-Eval scoring of 4 prompt variants + pairwise export |
| `evaluate_sensor_awareness.py` | Sensor awareness scoring across 11 scenarios |
| `human_ranking.py` | CLI pairwise ranking tool (ELO + win rates) |

---

## Workflow

### Step 1 — Run prompt variant evaluation

```bash
python evaluation/evaluate_prompt_variants.py --export-pairs
```

This generates two files:
- `results/prompt_variant_results.json` — automated scores per variant per question
- `results/prompt_pairs.json` — blinded A/B pairs for human ranking

### Step 2 — Human pairwise ranking

Each team member rates independently with their own rater ID:

```bash
python evaluation/human_ranking.py --rater-id romeo --n-pairs 20
python evaluation/human_ranking.py --rater-id ruben --n-pairs 20
```

View current standings at any time:

```bash
python evaluation/human_ranking.py --results
```

Ratings are saved incrementally — you can stop and resume at any time.

### Step 3 — Sensor awareness evaluation

```bash
python evaluation/evaluate_sensor_awareness.py
```

Results are saved to `results/sensor_awareness_results.json`.

Use `--verbose` to see the full answer text for each scenario:

```bash
python evaluation/evaluate_sensor_awareness.py --verbose
```

### Step 4 — Smoke tests (optional, before full runs)

```bash
# Test only 5 questions for prompt variants
python evaluation/evaluate_prompt_variants.py --n-questions 5

# Test only 4 sensor scenarios
python evaluation/evaluate_sensor_awareness.py --n-scenarios 4
```

---

## Dependencies

All scripts inherit the same environment as the rest of the evaluation pipeline. No additional packages needed beyond what is already in `requirements.txt`.

The sensor awareness evaluation uses mocked sensor data — no database connection or Raspberry Pi hardware is required.
