# ChickenCare AI - RAG System

Smart sensor integration + hybrid retrieval for chicken welfare advice.

## Key Features

1. **Smart Sensor Filtering** - Only includes relevant sensor data (reduces tokens by ~70%)
2. **Hybrid Retrieval** - Combines semantic (Chroma) + keyword (BM25) search
3. **PostgreSQL Integration** - Real sensor data from database
4. **Automated Critical Alerts** - Scheduler detects critical conditions, RAG generates advice
5. **Event Logging** - All LLM responses + sensor alerts logged for performance evaluation
6. **RAG vs NO-RAG Evaluation** - Measure improvement

---

## System Architecture

```
                         RASPBERRY PI                          DESKTOP SERVER
                    ┌─────────────────────┐             ┌──────────────────────────────┐
                    │  Sensors (DHT22 etc) │             │  PostgreSQL                  │
                    │         │            │             │  ┌────────────────────────┐  │
                    │         v            │   network   │  │  sensor_readings table │  │
                    │  pi_sensor_writer.py ├────────────>│  │  (temp, humidity, etc) │  │
                    │  (send_reading())    │   INSERT    │  └────────────────────────┘  │
                    └─────────────────────┘             │  ┌────────────────────────┐  │
                                                        │  │  event_log table       │  │
                                                        │  │  (queries, responses,  │  │
                                                        │  │   sensor snapshots)    │  │
                                                        │  └────────────────────────┘  │
                                                        └──────────────────────────────┘
                                                                     ^
                                                                     │ READ / WRITE
                                                                     v
                                                        ┌──────────────────────────────┐
                                                        │  FastAPI App (app.py)        │
                                                        │                              │
                                                        │  /ask    → RAG pipeline      │
                                                        │  /sensors → latest reading   │
                                                        │  /sensors/history → trends   │
                                                        │  /events  → event log        │
                                                        │  /        → React frontend   │
                                                        │                              │
                                                        │  scheduler.py (background)   │
                                                        │  checks sensors periodically │
                                                        │  critical? → triggers RAG    │
                                                        └──────────────────────────────┘
                                                                     ^
                                                                     │ HTTP (port 8000)
                                                                     v
                                                        ┌──────────────────────────────┐
                                                        │  React Web UI (frontend/)    │
                                                        │                              │
                                                        │  Chat tab  → /ask            │
                                                        │  Sensors   → /sensors        │
                                                        │  + charts  → /sensors/history│
                                                        │  Alerts    → /events         │
                                                        └──────────────────────────────┘
```

---

## Data Flow

### Flow 1: User Asks a Question (`/ask`)

```
User Question
    │
    v
Get latest sensor reading (PostgreSQL)
    │
    v
Should include sensors? (sensor_filter.py)
    │ - Critical alerts?     → always include
    │ - Environment query?   → include
    │ - General question?    → skip sensors
    v
Hybrid Retrieval (semantic + keyword) from knowledge base
    │
    v
Build prompt (context + sensor data if relevant)
    │
    v
LLM generates response (Ollama)
    │
    v
Log to event_log: query + answer + sensor snapshot + filtered context
    │
    v
Return answer to user
```

### Flow 2: Scheduler Detects Critical Conditions (automatic)

```
Scheduler runs every N minutes
    │
    v
Read latest sensor_readings row
    │
    v
get_critical_alerts() — anything critical?
    │
    ├─ NO  → was there a previous alert?
    │          YES → log "conditions_normal" to event_log
    │          NO  → do nothing
    │
    └─ YES → same alerts as last cycle?
               YES → skip (dedup, don't spam)
               NO  → build query from critical alerts
                        │
                        v
                     Run full RAG pipeline
                     (retrieval + LLM advice)
                        │
                        v
                     Log to event_log:
                       event_type = "sensor_alert"
                       user_query = "Critical coop conditions: ..."
                       llm_response = actionable advice
                       sensor_snapshot = raw reading
                       sensor_context_filtered = what the LLM saw
                        │
                        v
                     Frontend polls /events → shows alert + advice to user
```

### Flow 3: Sensor Team Writes Data (Raspberry Pi)

```
Pi reads sensors (DHT22, load cells, etc.)
    │
    v
Sensor team classifies: normal / warning / critical
(they decide the thresholds, not us)
    │
    v
pi_sensor_writer.send_reading(
    temperature_c=36.2,
    temperature_status="critical",
    humidity_pct=85,
    humidity_status="critical",
    ...
)
    │
    v
insert_sensor_reading() writes to PostgreSQL
over the network (Pi → Desktop via IP:port)
```

---

## Database Schema

### sensor_readings — raw data from the Pi

```sql
CREATE TABLE sensor_readings (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    temperature_c FLOAT,
    temperature_status TEXT,     -- 'normal', 'warning', 'critical' (set by sensor team)
    humidity_pct FLOAT,
    humidity_status TEXT,
    heat_stress_index TEXT,
    feeder_status TEXT,          -- 'full', 'low', 'empty'
    waterer_status TEXT
);
```

### event_log — system performance evaluation

```sql
CREATE TABLE event_log (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    event_type TEXT NOT NULL,            -- 'llm_response', 'sensor_alert', 'conditions_normal'
    severity TEXT DEFAULT 'info',         -- 'info', 'warning', 'critical'
    user_query TEXT,                      -- question asked (user or scheduler-generated)
    llm_response TEXT,                    -- full AI answer
    sensor_snapshot JSONB,                -- raw sensor data at time of event
    sensor_context_filtered TEXT          -- filtered context string passed to LLM
);
```

This lets us later review: what was asked, what sensor data was active, what the LLM saw (filtered), and what it answered.

---

## Files

```
├── app.py                    # FastAPI server — all API endpoints + serves frontend
├── db_utils.py               # PostgreSQL queries (sensor_readings + event_log)
├── sensor_filter.py          # Smart context filtering — decides when to include sensors
├── prompts.py                # System prompts for LLM (normal, emergency, simple)
├── rag_functions.py          # Complete RAG pipeline (hybrid retrieval + LLM)
├── scheduler.py              # Background: checks sensors, triggers RAG on critical
├── pi_sensor_writer.py       # Template for sensor team to push data from Pi
├── generate_demo_data.py     # Create test sensor readings
├── requirements.txt          # Python dependencies
├── .env.example              # Environment config template (copy to .env)
├── test_docs/                # Knowledge base (PDFs, TXTs)
├── chroma_db/                # Vector database (auto-created, gitignored)
├── frontend/                 # React web UI (Vite + Tailwind + Recharts)
│   ├── src/
│   │   ├── components/
│   │   │   ├── ChatPanel.jsx       # Chat tab — Q&A with LLM
│   │   │   ├── SensorDashboard.jsx # Live sensor cards + history chart
│   │   │   ├── SensorChart.jsx     # Recharts history chart (1h/24h/7d)
│   │   │   ├── AlertFeed.jsx       # Event log with expandable cards
│   │   │   └── Layout.jsx          # Top nav + tab switching
│   │   ├── api/index.js            # All fetch() calls centralised here
│   │   ├── App.jsx                 # Root component with tab state
│   │   └── main.jsx                # React entry point
│   ├── package.json
│   ├── vite.config.js              # Proxies /ask /sensors /events → :8000 in dev
│   └── dist/                       # Production build (gitignored, run npm run build)
└── evaluation/               # RAG evaluation scripts
    ├── evaluate_rag.py       # RAG vs NO-RAG heuristic comparison
    ├── evaluate_ragas.py     # RAGAS semantic evaluation
    ├── evaluate_deepeval.py  # DeepEval G-Eval metrics
    ├── evaluate_retrieval.py # Retrieval method comparison
    ├── evaluation_data.py    # Test questions + ground truth (fill in before running)
    ├── eval_config.py        # Scoring criteria (fill in before running)
    └── TEAMMATE_README.md    # Instructions for the evaluation content task
```

---

## Quick Start

### 1. Install Dependencies

```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt

ollama pull smollm2:1.7b
ollama pull nomic-embed-text-v2-moe:latest
```

### 2. Setup PostgreSQL

```bash
createdb chickens
# Configure .env with your DB credentials
```

### 3. Initialize Database + Demo Data

```bash
python db_utils.py
# Creates sensor_readings + event_log tables, inserts mock data
```

### 4. Add Knowledge Base

```bash
# Put chicken-keeping documents in test_docs/
cp /path/to/your/pdfs/*.pdf test_docs/
```

### 5. Test RAG Pipeline

```bash
python rag_functions.py
```

### 6. Build the Frontend

```bash
# Requires Node.js 18+ — download from https://nodejs.org
cd frontend
npm install
npm run build    # outputs to frontend/dist/
cd ..
```

### 7. Run the API Server

```bash
uvicorn app:app --reload
# Opens full UI at http://localhost:8000
# API docs at http://localhost:8000/docs
```

> **Dev mode (frontend hot-reload):** run `uvicorn app:app` in one terminal and
> `cd frontend && npm run dev` in another. The Vite dev server proxies API calls to `:8000`.

---

## Smart Sensor Context Filtering

```python
# Decision logic in sensor_filter.py:
if critical_sensor_alerts:
    include_in_prompt = True   # Always include critical
elif query_mentions_environment:
    include_in_prompt = True   # Relevant to question
elif resource_warning and query_mentions_resources:
    include_in_prompt = True   # Feeder/waterer query + low status
else:
    include_in_prompt = False  # Skip for general questions
```

When sensor context is included, `get_sensor_context()` reports only non-normal readings.
If all readings are currently normal it returns `"All readings normal."` — so the LLM always
gets a direct answer to environment questions rather than silence.

Readings older than 30 minutes are treated as stale and excluded regardless of query type.

| Query | Sensor Status | Include? | Context sent to LLM |
|-------|--------------|----------|---------------------|
| "How often do chickens lay eggs?" | Normal | No | _(none)_ |
| "Is it too hot in the coop?" | Normal | Yes | "All readings normal." |
| "Why are my chickens panting?" | Temp warning | Yes | Elevated temperature reading |
| "What breed should I get?" | Temp critical | Yes | Critical alert (always shown) |
| "Is the feeder full?" | Feeder low | Yes | Feeder: Low |

---

## Hybrid Retrieval

```
Semantic Search (60%)  → Understands "heat stress" concept
    +
Keyword Search (40%)  → Catches exact terms like "ammonia ppm"
    =
Better Results (evaluated via evaluate_rag.py — RAG vs NO-RAG)
```

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Serves React frontend (production build) |
| GET | `/api/info` | API info + endpoint listing (JSON) |
| GET | `/health` | Health check |
| POST | `/ask` | Ask a question (RAG pipeline) |
| GET | `/sensors` | Latest sensor reading |
| GET | `/sensors/history` | Historical readings (`?range=1h\|24h\|7d`) |
| GET | `/events` | Event log (LLM responses + sensor alerts) |
| POST | `/setup-db` | Create database tables |

---

## Configuration

### Database Connection (.env)

```
DB_HOST=localhost
DB_PORT=5432
DB_NAME=chickens
DB_USER=postgres
DB_PASSWORD=your_password
```

### Scheduler Interval

```
SCHEDULER_INTERVAL=60    # seconds (default 60 s)
```

### Simulation Mode

When the Raspberry Pi is not yet available, set `SIMULATION_MODE=true` to have the
server insert a synthetic sensor reading every 60 seconds — allowing you to test the
full pipeline (alerts, RAG advice, event log, charts) without real hardware.

```
SIMULATION_MODE=true     # fake Pi readings every 60 s
SIMULATION_MODE=false    # (default) live mode — only real Pi data
```

Scenario follows the time of day (cool night → normal morning → warm midday → hot/critical
afternoon), with an ~8 % random chance of resource_low regardless of hour.
To switch back to live mode: set `SIMULATION_MODE=false` (or remove it) and restart uvicorn.

### LLM Model

```
OLLAMA_MODEL=smollm2:1.7b
OLLAMA_EMBED_MODEL=nomic-embed-text-v2-moe:latest
```

---

## Methods (for paper)

### Retrieval-Augmented Generation (RAG) pipeline

The system uses a hybrid retrieval strategy combining dense vector search with sparse keyword
search. At query time both retrievers run in parallel and their results are merged via
**Reciprocal Rank Fusion (RRF)** through LangChain's `EnsembleRetriever` with weights
0.6 (semantic) / 0.4 (keyword).

**Embedding model**: `nomic-embed-text-v2-moe:latest` (via Ollama) — a mixture-of-experts
sentence embedding model chosen for strong retrieval performance at low resource cost.

**Semantic retriever**: Chroma vector store with HNSW index. Uses **Maximum Marginal
Relevance (MMR)** sampling (`fetch_k = 9`, `k = 3`) to penalise redundant chunks and
promote diversity in the retrieved context.

**Keyword retriever**: BM25 (Okapi BM25) via LangChain's `BM25Retriever` (`k = 3`).
Captures exact terminology (e.g. species names, specific symptoms) that dense embeddings
may underweight. The BM25 index is built **in-memory from the same chunks as Chroma** at
startup — it is a completely separate structure and does **not** query the vector store.
`k = 3` is the number of top chunks returned, not the BM25 k₁ saturation factor; k₁
defaults to 1.5 inside `rank_bm25.BM25Okapi`, which is within the recommended 1.2–1.5
range and is left at its default.

### Document preprocessing

- Source documents: `.txt` and `.pdf` files in the knowledge base folder.
- **PDF loading**: PDFs are converted to Markdown via `pymupdf4llm` before chunking
  (replaces `PyPDFLoader`). `PyPDFLoader` strips all document structure — headers,
  bullet points, and table rows become flat unformatted text, causing chunks to lose the
  heading that explained their context (e.g. "38 °C is dangerous" losing the "heat stress"
  heading). Markdown conversion preserves that structure, giving retrieved chunks richer
  context for the LLM.
- Splitting: `RecursiveCharacterTextSplitter` with chunk size 600 chars, overlap 100 chars,
  separators `["\n\n", "\n", ". ", " ", ""]` (paragraph → sentence → word boundary order).
- Post-split filtering: chunks shorter than 50 characters are discarded to remove headers,
  page numbers, and OCR artefacts that add noise without information.
- The vector store is rebuilt automatically when the knowledge-base fingerprint (MD5 of
  filenames + modification timestamps) changes.

### Context budget and prompt design

The LLM (`smollm2:1.7b`) has a limited context window, so token
budget is managed explicitly:
- Each retrieved chunk is capped at 600 characters (aligned with chunk size).
- `format_context()` enforces a hard ceiling of **3 000 characters** total context
  (≈ 750 tokens at ~4 chars/token). Once the ceiling is reached, additional chunks are
  silently dropped. This guard prevents silent quality degradation if the knowledge base
  grows or `k` is changed — without it the model's input can overflow the context window
  and produce truncated or degraded responses with no warning.
- Up to 5 chunks are retrieved (`k=5`); the budget guard limits how many are actually sent.
- Sensor context is injected only when relevant (see Smart Sensor Context Filtering above),
  preventing unnecessary token use on general queries.

Three prompt templates are selected at runtime:
- **SYSTEM_PROMPT** — general query with sensor context.
- **BASIC_EMERGENCY_PROMPT** — environment query with critical sensor alerts; structured
  output (situation / immediate actions / when to call a vet).
- **SIMPLE_PROMPT** — no sensor context needed.

### LLM configuration

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Model | `smollm2:1.7b` (local, via Ollama) | Runs fully offline on consumer hardware (~1.1 GB RAM). SmolLM2 is a HuggingFace model reported to outperform same-size Qwen2.5 on instruction-following benchmarks. Selected over `qwen2.5:1.5b-instruct` after the larger `gemma3n:e2b` (5.8 GB) exceeded available memory on the development machine. |
| Temperature | 0.1 | Low temperature for consistency and factual grounding — a safety-critical advisory system must give the same answer to the same question; high temperature introduces random variation and increases hallucination risk |
| Max new tokens | 400 | Sufficient for a complete actionable answer without padding |
| Embedding model | `nomic-embed-text-v2-moe:latest` | Mixture-of-experts sentence embedding model; strong retrieval performance, runs locally via Ollama |

### Sensor integration

The sensor team (Raspberry Pi) classifies each reading as `normal / warning / critical`
before writing to PostgreSQL. This system consumes those labels without re-defining
thresholds, separating concerns between data collection and advisory logic.

Readings older than 30 minutes are treated as stale and excluded from both the RAG
pipeline and scheduler checks to prevent advice based on outdated conditions.

### Evaluation

`evaluate_rag.py` implements a deterministic evaluation comparing RAG responses against
a no-retrieval baseline (same LLM, same question, no knowledge context). Metrics:
- **Topic coverage**: fraction of expected keywords present in the answer.
- **Length appropriateness**: scored against a 100–300 word target range.
- **Actionability**: presence of numbered steps or action-oriented language.

All queries and LLM responses are logged to the `event_log` table with the exact
sensor snapshot and filtered context the LLM received, enabling post-hoc performance
analysis.

---

## For Your Presentation

### Key Points

1. **Problem:** LLMs hallucinate without grounding in real knowledge
2. **Solution:** RAG retrieves relevant docs before generating answer
3. **Innovation:** Smart sensor filtering (only include when needed)
4. **Automation:** Scheduler detects critical conditions, RAG generates advice automatically
5. **Evaluation:** All responses logged to event_log for performance review
6. **Result:** Better accuracy with manageable latency

### Demo Script

1. Show normal query: "How often do chickens lay eggs?"
   - No sensors included, fast response

2. Show environmental query: "Is it too hot in the coop?"
   - Sensors always included for environment queries
   - If readings are normal: LLM responds with "All readings normal" confirmation
   - If readings are elevated: LLM responds with current values and advice

3. Show critical scenario (insert critical reading into DB):
   - Scheduler triggers, RAG generates emergency advice
   - Check `/events` to see the logged alert + advice

4. Run evaluation:
   - Show RAG beats NO-RAG, quantify improvement

---

## Notes

- Sensor team provides status labels (normal/warning/critical) — we don't define thresholds
- PostgreSQL is recommended over JSON (faster, safer, scalable)
- All code is well-commented for biosystems students
- event_log stores everything needed to evaluate system performance later
