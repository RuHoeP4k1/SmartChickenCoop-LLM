# ChickenCare AI

A RAG-based question-answering system for poultry farmers. Combines a hybrid retrieval knowledge base with live coop sensor data to provide contextual, actionable advice. Built as a research project targeting a scientific report on LLM configuration, prompt design, and sensor-aware context injection.

---

## Architecture

```
Raspberry Pi ──serial──> sensor_readings (PostgreSQL)
                                │
                                ▼
User ──> React (ChatPanel) ──> POST /ask ──> app.py
                                              ├─ sensor_filter.py  → include sensors? (LLM semantic routing)
                                              ├─ rag_functions.py  → hybrid retrieval (60% semantic / 40% BM25, k=4)
                                              ├─ prompts.py        → select SIMPLE / MAIN / EMERGENCY (hybrid design)
                                              ├─ LLM inference     → ministral-14b-2512 via OpenRouter
                                              ├─ db_utils.py       → log to event_log
                                              └─ return {answer, sources, sensor_included, has_critical}

APScheduler ──> check_sensors() every N sec ──> critical? ──> RAG ──> event_log (sensor_alert)
```

---

## Stack

| Layer | Technology |
|-------|------------|
| Backend | FastAPI (Python 3.11), uvicorn |
| Database | PostgreSQL (Supabase hosted or local), psycopg2 connection pool |
| Vector store | ChromaDB (persisted) + BM25 (in-memory) via LangChain EnsembleRetriever |
| LLM | Ministral-14b-2512 via OpenRouter (`openrouter/mistralai/ministral-14b-2512`) |
| Embeddings | nomic-embed-text-v2-moe (local Ollama) |
| Scheduling | APScheduler — monitors sensors, triggers RAG alerts |
| Frontend | React 18 + Vite + Tailwind CSS + Recharts |
| Evaluation | DeepEval (G-Eval), RAGAS, heuristic scorers, human pairwise ranking |
| Container | Docker multi-stage (Node build → Python runtime) |

---

## Quick Start

### Dev (two terminals)

```bash
# Backend
pip install -r requirements.txt
cp .env.example .env   # fill in DB credentials + API keys
uvicorn app:app --reload

# Frontend
cd frontend
npm install
npm run dev   # http://localhost:5173 — proxies /ask /sensors /events to :8000
```

### Production

```bash
cd frontend && npm run build
uvicorn app:app   # serves frontend + API at http://localhost:8000
```

### Docker

```bash
docker compose up --build   # first run (~20 min to embed docs)
docker compose up           # subsequent runs (instant)
```

### Database reset

```bash
psql -U postgres -d chickens -c "TRUNCATE sensor_readings, event_log RESTART IDENTITY;"
python scripts/generate_demo_data.py
```

---

## Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `DATABASE_URL` | No | — | Full Supabase connection string (port 6543). If blank, uses `DB_*` vars. |
| `DB_HOST` / `DB_PORT` / `DB_NAME` / `DB_USER` / `DB_PASSWORD` | Yes (if no DATABASE_URL) | — | Local PostgreSQL credentials |
| `OLLAMA_MODEL` | Yes | `smollm2:1.7b` | LLM model. Prefix `openrouter/` to route via OpenRouter. Production: `openrouter/mistralai/ministral-14b-2512` |
| `OLLAMA_EMBED_MODEL` | Yes | `nomic-embed-text-v2-moe:latest` | Embedding model (local Ollama). Change → delete `chroma_db/` to re-embed. |
| `OPENROUTER_API_KEY` | For OpenRouter models | — | Required when `OLLAMA_MODEL` starts with `openrouter/` |
| `ANTHROPIC_API_KEY` | For eval | — | Claude Haiku judge for G-Eval / RAGAS |
| `SIMULATION_MODE` | No | `false` | `true` = fake sensor readings every 60 s. Use `=true` not `=#true`. |
| `SCHEDULER_INTERVAL` | No | 60 (sim) / 600 (live) | Seconds between sensor checks |
| `CHAT_HISTORY_TURNS` | No | `2` | Past Q&A turns included in prompt context |
| `COOP_LAT` / `COOP_LON` | No | `50.8798` / `4.7005` | Coordinates for Open-Meteo weather API |
| `SENSOR_ROUTER_MODEL` | No | → `OLLAMA_MODEL` | Override model for LLM-based sensor routing only |

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/ask` | RAG pipeline — returns `{answer, sources, sensor_included, has_critical}` |
| GET | `/sensors` | Latest sensor reading |
| GET | `/sensors/history` | Historical readings (`?range=1h\|24h\|7d`) |
| GET | `/events` | Event log (LLM responses + scheduler alerts) |
| GET | `/weather` | Open-Meteo forecast for coop location |
| GET | `/health` | Health check |
| POST | `/setup-db` | Create database tables |
| GET | `/api/info` | API info + endpoint listing |

---

## RAG Pipeline

### Retrieval

Hybrid retrieval via LangChain `EnsembleRetriever` (Phase 1 sweep winner config):
- **Semantic** (70%) — ChromaDB cosine similarity via nomic-embed-text-v2-moe embeddings
- **BM25** (30%) — exact term matching, ~0 ms, catches specific terminology

Documents are chunked at 1000 characters with 120-char overlap. Chunks shorter than 50 characters are discarded. Context budget: 8000 characters max passed to the LLM.

### Sensor Context Injection

`sensor_filter.py` decides whether to inject live sensor readings into the prompt using **semantic LLM routing** (default).

**LLM classifier** (`llm_route_sensors`) — Phase 2 winner, production default:
- Single-turn prompt to ministral-14b-2512, returns `INCLUDE` or `EXCLUDE`
- Achieves **100% accuracy** (19/19 scenarios) vs keyword filter's 94.7% (18/19)
- Latency: ~400–1100 ms | Cost: ~$0.00009/call
- Falls back to keyword filter on parse error
- Configurable via `SENSOR_ROUTER_MODEL` env var

**Keyword filter** (`should_include_sensors`) — fallback, ~0 ms, no cost:
- 7 keyword sets covering situational queries, symptoms, air quality, resources, door/flock/egg status
- Priority-ordered rules; `_GENERAL_KNOWLEDGE_SIGNALS` suppresses injection for encyclopedic questions
- Enabled by setting `SENSOR_ROUTING_MODE=keyword`

Readings older than 30 minutes are treated as stale and excluded.
`get_sensor_context()` reports only non-normal readings. Normal state returns `"All coop readings normal."` rather than silence.

### Prompt Selection

| Condition | Template | Design |
|-----------|----------|--------|
| Sensor routing decides EXCLUDE | `SIMPLE_PROMPT` | Structured (short answer → steps → vet decision) |
| Sensor routing decides INCLUDE, no critical | `MAIN_PROMPT` | Sensor data validates/prioritises knowledge base claims |
| Critical sensor alerts present | `EMERGENCY_PROMPT` | Critical reading leads; knowledge base supports actions only |

**Hybrid Prompt Design** (Phase 2, 2026-03-27):
- Combines structured layout (actionability) + conciseness (~800 tokens) + expert authority
- Anticipates follow-up questions; does not pre-answer everything
- Sensor data acts as specificity amplifier (validates which knowledge base claims are most relevant *right now*)
- Format: markdown with bold, bullets, clear sections

LLM output budget: 400 tokens max.

---

## Evaluation

### Phase 1 — RAG Configuration Sweep (complete)

Full-factorial sweep: 3 LLMs × 2 chunk sizes × 2 k values × 2 hybrid weight configs = 24 configurations, 30 questions each, scored by Llama 3.3 70B via OpenRouter (7 metrics: actionability, correctness, faithfulness, answer relevancy, contextual precision/recall/relevancy).

**Winner:** `ministral-14b-2512` | chunk=1000 | k=4 | weights=[0.7, 0.3] (70% semantic / 30% BM25)
**Combined score:** 0.9667 (actionability: 0.990, correctness: 0.943)

**Key finding:** LLM model is the only statistically significant factor (ANOVA F=34.85, p<0.0001). Chunk size and k are significant for retrieval quality but not for answer quality at this scale.

See `evaluation/SWEEP_README.md` and `evaluation/results/round1_analysis.md`.

### Phase 2 — Prompt Design + Sensor Awareness (complete)

**Prompt variants** (complete, 2026-03-27): 4 variants (baseline, structured, concise, expert) evaluated via:
- **G-Eval scoring** (DeepEval): Actionability + Correctness via Kimi 2.6 on 30 questions
- **Human pairwise ranking**: 37 raters, 644 votes, 178 pairs rated (99% coverage)
- **Mixed effects analysis**: random intercept per question to isolate variant effects

Winner: **`structured` variant** — 56.1% win rate (human), best actionability score (0.890 vs 0.827 baseline). New production prompt design combines all four variants' strengths (hybrid design).

**Sensor routing comparison**: keyword filter vs LLM classifier on 19 scenarios:
- **Keyword filter**: 18/19 (94.7%) — fails on S13 (H₂S critical + encyclopedic question)
- **LLM classifier**: 19/19 (100%) — correctly reasons about question type vs sensor state
- Production: **LLM routing is now default** (`SENSOR_ROUTING_MODE=llm`)

**Sensor awareness end-to-end**: 19 scenarios across all sensor types.
Pass rate: **16/19 (84.2%)**. Three failure modes: over-hedging on normal (S03), urgency tone on encyclopedic (S11), H₂S critical overriding encyclopedic rule (S13 — now fixed by LLM routing).

See `evaluation/PHASE2_EVALUATION.md` for full results, human ranking stats, and qualitative feedback.

---

## Project Structure

```
├── app.py                         FastAPI server — all endpoints + frontend serving
├── backend/
│   ├── rag_functions.py           Hybrid retrieval + LLM inference
│   ├── sensor_filter.py           Sensor context injection (keyword + LLM routing)
│   ├── prompts.py                 SIMPLE / MAIN / EMERGENCY prompt templates
│   ├── db_utils.py                PostgreSQL queries (sensor_readings, event_log)
│   └── scheduler.py              APScheduler — monitors sensors, triggers alerts
├── scripts/
│   └── generate_demo_data.py      Generates demo sensor readings (5 scenarios)
├── test_docs/                     Knowledge base source documents (PDF/TXT)
├── chroma_db/                     Persisted ChromaDB vector store (gitignored)
├── frontend/
│   └── src/
│       ├── App.jsx                Root component (TABS array)
│       ├── api/index.js           All fetch() calls
│       └── components/
│           ├── ChatPanel.jsx      Chat tab — Q&A with LLM
│           ├── SensorDashboard.jsx Live sensor cards
│           ├── SensorChart.jsx    Recharts history (1h/24h/7d)
│           ├── AlertFeed.jsx      Event log
│           ├── EggCalendar.jsx    Egg collection calendar
│           ├── AutomationPanel.jsx Automation rules
│           ├── Weather.jsx        Open-Meteo forecast
│           ├── MyChickens.jsx     Flock management
│           └── Layout.jsx         Top nav + tab switching
└── evaluation/
    ├── sweep.py                   Hyperparameter sweep runner
    ├── sweep_config.py            Parameter grid definition
    ├── sweep_analysis.py          ANOVA + main effects analysis
    ├── sweep_mixed_model.py       Linear mixed model (per-question random effects)
    ├── evaluate_prompt_variants.py Prompt design G-Eval scoring
    ├── evaluate_sensor_awareness.py 19-scenario sensor awareness test
    ├── compare_sensor_routing.py  Keyword vs LLM routing comparison
    ├── human_ranking.py           CLI pairwise ELO ranking tool
    ├── evaluation_data.py         30 test questions + ground truth
    ├── eval_config.py             G-Eval scoring criteria
    ├── EVALUATION_OVERVIEW.md     Metrics reference + current status
    ├── SWEEP_README.md            Sweep design + results
    ├── PHASE2_EVALUATION.md       Phase 2 design + results
    └── results/
        ├── doe_design.md          Design matrix (24 runs)
        ├── round1_analysis.md     Sweep results (auto-generated)
        ├── mixed_model_analysis.md Mixed-effects model output
        ├── prompt_variant_results.json G-Eval scores per variant
        └── sensor_awareness_results.json 19-scenario pass/fail results
```

---

## Database Schema

```sql
-- Live sensor data from Raspberry Pi
CREATE TABLE sensor_readings (
    id                  SERIAL PRIMARY KEY,
    timestamp           TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    temperature_c       FLOAT,
    temperature_status  TEXT,   -- 'normal' | 'warning' | 'critical'
    humidity_pct        FLOAT,
    humidity_status     TEXT,
    heat_stress_index   TEXT,
    feeder_status       TEXT,   -- 'full' | 'low' | 'empty'
    feeder_pct          FLOAT,
    waterer_status      TEXT,
    waterer_pct         FLOAT,
    h2s_level           TEXT,
    h2s_ppm             FLOAT,
    mold_risk_status    TEXT,
    crowding_assessment TEXT,
    door_open           BOOLEAN,
    number_of_chickens  INTEGER,
    egg_count           INTEGER
);

-- All LLM responses and scheduler alerts
CREATE TABLE event_log (
    id                       SERIAL PRIMARY KEY,
    timestamp                TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    event_type               TEXT NOT NULL,  -- 'llm_response' | 'sensor_alert' | 'conditions_normal'
    severity                 TEXT DEFAULT 'info',
    user_query               TEXT,
    llm_response             TEXT,
    sensor_snapshot          JSONB,
    sensor_context_filtered  TEXT
);
```

Status labels (`normal / warning / critical`) are set by the sensor team on the Raspberry Pi — this system consumes them without redefining thresholds.

---

## Production Deployment (Render)

### Steps

1. Push your code to GitHub.
2. Go to [render.com](https://render.com) → **New Web Service** → connect your GitHub repo.
3. Set **Build Command**: `cd frontend && npm install && npm run build`
4. Set **Start Command**: `uvicorn app:app --host 0.0.0.0 --port $PORT`
5. Under **Environment**, add all required variables from `.env.example` (see [Environment Variables](#environment-variables) above).
6. Click **Deploy**.

### Free tier spin-down

Render's free tier pauses the service after ~15 minutes of inactivity. Open `https://your-app.onrender.com/health` in a browser **at least 1 minute before a demo** to wake it up.

### Custom domain

1. Render dashboard → your service → **Settings** → **Custom Domains** → add your domain.
2. In your DNS provider, add a `CNAME` record pointing your domain to the value Render shows (e.g. `your-app.onrender.com`).

### Generate a secure API key

```bash
python -c "import secrets; print(secrets.token_urlsafe(32))"
```
