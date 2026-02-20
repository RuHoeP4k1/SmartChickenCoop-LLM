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
                                                        │  /events  → event log        │
                                                        │                              │
                                                        │  scheduler.py (background)   │
                                                        │  checks sensors periodically │
                                                        │  critical? → triggers RAG    │
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
├── app.py                    # FastAPI server — /ask, /sensors, /events endpoints
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

ollama pull qwen2.5:1.5b-instruct
ollama pull nomic-embed-text
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

### 6. Run the API Server

```bash
uvicorn app:app --reload
# API docs at http://localhost:8000/docs
```

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
| GET | `/` | API info + endpoint listing |
| GET | `/health` | Health check |
| POST | `/ask` | Ask a question (RAG pipeline) |
| GET | `/sensors` | Latest sensor reading |
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
SCHEDULER_INTERVAL=900    # seconds (default 15 min)
```

### LLM Model

```
OLLAMA_MODEL=qwen2.5:1.5b-instruct
OLLAMA_EMBED_MODEL=nomic-embed-text
```

---

## Methods (for paper)

### Retrieval-Augmented Generation (RAG) pipeline

The system uses a hybrid retrieval strategy combining dense vector search with sparse keyword
search. At query time both retrievers run in parallel and their results are merged via
**Reciprocal Rank Fusion (RRF)** through LangChain's `EnsembleRetriever` with weights
0.6 (semantic) / 0.4 (keyword).

**Embedding model**: `nomic-embed-text` (768-dimensional, via Ollama) — a sentence-level
embedding model chosen for strong retrieval performance at low resource cost.

**Semantic retriever**: Chroma vector store with HNSW index. Uses **Maximum Marginal
Relevance (MMR)** sampling (`fetch_k = 9`, `k = 3`) to penalise redundant chunks and
promote diversity in the retrieved context.

**Keyword retriever**: BM25 (Okapi BM25) via LangChain's `BM25Retriever` (`k = 3`).
Captures exact terminology (e.g. species names, specific symptoms) that dense embeddings
may underweight.

### Document preprocessing

- Source documents: `.txt` and `.pdf` files in the knowledge base folder.
- Splitting: `RecursiveCharacterTextSplitter` with chunk size 600 chars, overlap 100 chars,
  separators `["\n\n", "\n", ". ", " ", ""]` (paragraph → sentence → word boundary order).
- Post-split filtering: chunks shorter than 50 characters are discarded to remove headers,
  page numbers, and OCR artefacts that add noise without information.
- The vector store is rebuilt automatically when the knowledge-base fingerprint (MD5 of
  filenames + modification timestamps) changes.

### Context budget and prompt design

The 1.5B-parameter LLM (`qwen2.5:1.5b-instruct`) has a limited context window, so token
budget is managed explicitly:
- Each retrieved chunk is capped at 600 characters (aligned with chunk size).
- Truncation falls back to the nearest sentence boundary to avoid mid-sentence cuts.
- Up to 3 chunks are included (≈1 800 chars of knowledge context).
- Sensor context is injected only when relevant (see Smart Sensor Context Filtering above),
  preventing unnecessary token use on general queries.

Three prompt templates are selected at runtime:
- **SYSTEM_PROMPT** — general query with sensor context.
- **BASIC_EMERGENCY_PROMPT** — environment query with critical sensor alerts; structured
  output (situation / immediate actions / when to call a vet).
- **SIMPLE_PROMPT** — no sensor context needed.

### LLM configuration

| Parameter | Value |
|-----------|-------|
| Model | `qwen2.5:1.5b-instruct` (local, via Ollama) |
| Temperature | 0.7 |
| Max new tokens | 400 |
| Embedding model | `nomic-embed-text` |

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
