# Phase 2 Evaluation Plan

Once the configuration sweep (model / chunk_size / k / weights) is complete, two independent evaluation axes remain.

---

## Axis 1 — Prompt design

**Goal:** find which prompt template produces the best answers, holding the RAG configuration constant.

Four variants of the `SIMPLE_PROMPT` are tested (the most-used path — no live sensor context):

| Variant | Design choice |
|---|---|
| `baseline` | Current production prompt |
| `structured` | Forces numbered output (Short answer / What to do / Call a vet if) |
| `concise` | Minimal instructions — relies on the model's own judgement |
| `expert` | Positions the assistant as a poultry scientist |

Evaluation is done in two ways:

1. **Automated heuristic scoring** — topic coverage, length appropriateness, actionability (same metrics as `evaluate_rag.py`).
2. **Human pairwise preference** — blinded A vs B comparisons where raters don't know which variant they're seeing. Results are aggregated into win rates and ELO scores. This human evaluation method can be extended beyond 
only evaluating the different prompt variants ofc!

---

## Axis 2 — Sensor / real-time awareness

**Goal:** verify that the system correctly integrates live sensor data, independent of RAG retrieval quality.

Even with a perfect knowledge base, the system needs to:

1. **Include** sensor context when the reading is relevant to the question
2. **Exclude** sensor context for general knowledge questions with normal readings
3. **Cite** the actual sensor values (e.g. "35.2°C" should appear in the answer)
4. **Match urgency to severity** — critical readings → urgent language; normal readings → calm tone
5. **Not fabricate** sensor values when no sensor context was injected into the prompt

Eleven test scenarios cover these cases using mocked sensor snapshots (no Pi or database needed):

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

---

## Files

| File | Purpose |
|---|---|
| `evaluate_prompt_variants.py` | Automated scoring of 4 prompt variants + pairwise export |
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
