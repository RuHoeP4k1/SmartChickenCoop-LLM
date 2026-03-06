# COMPLETE SETUP GUIDE - ChickenCare AI RAG System

**Follow these steps EXACTLY. Don't skip anything!**

---

## Step 1: Install PostgreSQL

### Windows:
```bash
# Download from: https://www.postgresql.org/download/windows/
# Run installer, remember your password!
# Default settings are fine

# After install, add to PATH (if not automatic):
# System Properties > Environment Variables > Path > Add:
# C:\Program Files\PostgreSQL\16\bin
```

### Mac:
```bash
brew install postgresql@16
brew services start postgresql@16
```

### Linux (Ubuntu):
```bash
sudo apt update
sudo apt install postgresql postgresql-contrib
sudo service postgresql start
```

### Verify PostgreSQL is running:
```bash
# All platforms
psql --version
# Should show: psql (PostgreSQL) 16.x
```

---

## Step 2: Create Database

```bash
# Create the database
createdb chickens

# If that doesn't work, try:
# Windows: createdb -U postgres chickens
# Linux: sudo -u postgres createdb chickens

# Test connection:
psql -d chickens
# You should see: chickens=#
# Type \q to exit
```

---

## Step 3: Install Ollama + Models

### Windows/Mac:
```bash
# Download from: https://ollama.com/download
# Run installer

# After install, pull models:
ollama pull qwen2.5:1.5b-instruct
ollama pull nomic-embed-text

# Verify:
ollama list
# Should show both models
```

### Linux:
```bash
curl -fsSL https://ollama.com/install.sh | sh
ollama pull qwen2.5:1.5b-instruct
ollama pull nomic-embed-text
```

---

## Step 4: Install Node.js (for the Web UI)

### Windows / Mac:
```bash
# Download the LTS installer from: https://nodejs.org
# Run installer — default settings are fine

# Verify:
node --version   # should show v18 or higher
npm --version
```

### Linux (Ubuntu):
```bash
curl -fsSL https://deb.nodesource.com/setup_20.x | sudo -E bash -
sudo apt install -y nodejs
```

---

## Step 4b: Setup Python Environment

```bash
# Navigate to your project folder
cd /path/to/your/project

# Create virtual environment
python -m venv venv

# Activate it
# Windows:
venv\Scripts\activate

# Mac/Linux:
source venv/bin/activate

# You should see (venv) in your terminal now
```

---

## Step 5: Install Python + Frontend Dependencies

```bash
# Make sure venv is activated!
pip install -r requirements.txt

# This will install:
# - psycopg2-binary (PostgreSQL)
# - langchain + chromadb (RAG)
# - pypdf (document loading)
# - fastapi (API server)

# Should take 2-3 minutes

# Install frontend dependencies
cd frontend
npm install
cd ..
```

---

## Step 6: Configure Database Connection

Create or edit a `.env` file in the project root:

```
DB_HOST=localhost
DB_PORT=5432
DB_NAME=chickens
DB_USER=postgres
DB_PASSWORD=your_password
OLLAMA_MODEL=qwen2.5:1.5b-instruct
OLLAMA_EMBED_MODEL=nomic-embed-text
```

**Windows default:** user is usually `postgres`
**Mac/Linux:** might be your system username

---

## Step 7: Create Folder for Knowledge Base

```bash
# Create folder for your chicken-keeping documents
mkdir test_docs

# Add some PDFs or TXT files about chickens
# Can be research papers, care guides, anything!
# At least 1 file needed to start
```

**Don't have documents yet?** Create a simple one:

```bash
echo "Chickens need temperatures between 20-24C for optimal laying. High temperatures above 30C can cause heat stress. Provide plenty of water and shade." > test_docs/chicken_basics.txt
```

---

## Step 8: Generate Demo Sensor Data

```bash
# Run the demo data generator
python generate_demo_data.py

# Choose option 1 (simple test data) first time
# This will:
#   1. Create the sensor_readings + event_log tables
#   2. Insert 5 test readings
#   3. Show you the latest reading
```

---

## Step 9: Build the Web UI

```bash
cd frontend
npm run build
# Outputs production files to frontend/dist/
# FastAPI will serve these automatically on http://localhost:8000
cd ..
```

> **Dev mode (optional):** run `npm run dev` instead to get Vite hot-reload
> while developing the UI. API calls are proxied to FastAPI on port 8000.

---

## Step 10: Test the RAG Pipeline

```bash
python rag_functions.py

# First run will be SLOW (2-5 minutes)
# - Loads documents from test_docs/
# - Creates embeddings
# - Builds vector database in chroma_db/

# Second run will be FAST (<1 second)
# - Just loads existing chroma_db/
```

---

## Step 11: Run Evaluation (RAG vs NO-RAG)

All evaluation scripts are run from the project root. Results are saved to `evaluation/results/`.

```bash
# Fast heuristic scoring — RAG vs NO-RAG (5-10 min, no ground truth needed)
python evaluation/evaluate_rag.py

# G-Eval custom metrics via DeepEval (actionability + correctness, no ground truth needed)
python evaluation/evaluate_deepeval.py

# RAGAS semantic metrics — faithfulness, relevancy, precision, recall
# Requires ground_truth filled in evaluation/evaluation_data.py first
python evaluation/evaluate_ragas.py

# Hybrid vs semantic-only retrieval comparison (also needs ground_truth)
python evaluation/evaluate_retrieval.py
```

---

## Step 12: Test Pi Sensor Writer

```bash
python pi_sensor_writer.py

# This verifies the database connection and inserts a test reading.
# The sensor team copies this file onto their Raspberry Pi and sets
# DB_HOST to the desktop's IP address in their .env file.
```

**Expected output:**
```
Pi Sensor Writer - Connection Test
========================================
Connection OK!

Inserting a test reading...
Inserted row id=4

You're good to go. Use send_reading() in your sensor loop.
```

---

## Quick Reference Commands

```bash
# ── Frontend ──────────────────────────────────────────────────────
# Build for production (required before uvicorn serves the UI)
cd frontend && npm run build && cd ..

# Dev mode — Vite hot-reload (open http://localhost:5173)
# Run in a separate terminal while uvicorn is running
cd frontend && npm run dev

# ── Backend ───────────────────────────────────────────────────────
# Start the API + serve the built frontend
uvicorn app:app --reload
# → open http://localhost:8000

# ── Database ──────────────────────────────────────────────────────
# Generate new demo data
python generate_demo_data.py

# Check latest sensor reading
python -c "from db_utils import get_latest_sensor_reading; print(get_latest_sensor_reading())"

# Reset all sensor + event data (keeps tables)
python -c "
import psycopg2, os
from dotenv import load_dotenv
load_dotenv()
conn = psycopg2.connect(host=os.getenv('DB_HOST'), dbname=os.getenv('DB_NAME'),
                        user=os.getenv('DB_USER'), password=os.getenv('DB_PASSWORD'))
conn.autocommit = True
conn.cursor().execute('TRUNCATE sensor_readings, event_log RESTART IDENTITY')
print('Done')
"

# ── RAG ───────────────────────────────────────────────────────────
# Test RAG pipeline
python rag_functions.py

# Rebuild vector database (after adding new documents)
rm -rf chroma_db/      # Windows: rmdir /s /q chroma_db
python rag_functions.py

# Run evaluation
python evaluation/evaluate_rag.py

# Test Pi sensor writer
python pi_sensor_writer.py
```

---

## Simulation Mode (Testing Without a Pi)

When the Raspberry Pi is not yet available you can run the full system end-to-end using
**simulation mode**. The server will insert one synthetic sensor reading every 60 seconds,
the scheduler will check it and fire alerts/RAG advice as normal, and the UI charts will
update automatically.

### Enable

Add one line to your `.env` file:

```
SIMULATION_MODE=true
```

Then restart uvicorn. You'll see this in the logs:
```
SIMULATION MODE ON — fake readings every 60 s, scheduler every 60s
```

### What you can test

| Feature | How to observe |
|---------|---------------|
| Live sensor cards updating | Sensors tab — values change every ~60 s |
| Historical charts filling in | Chart auto-refreshes; switch 1h / 24h / 7d |
| Automatic critical alert + RAG advice | Wait a few minutes — a "critical" reading will trigger and appear in the Alerts tab |
| Conditions-normal log entry | After a critical clears, scheduler logs "All Clear" |
| Q&A with active sensor context | Ask "Is it too hot?" during a hot/critical period |

### Scenario schedule

Readings follow the **time of day** (same pattern as the chicken coop's real environment):

| Time | Scenario |
|------|---------|
| 22:00 – 06:00 | `cold_night` — temp warning, humidity normal |
| 06:00 – 10:00 | `normal` — all readings normal |
| 10:00 – 14:00 | `hot_day` — temp + humidity warning |
| 14:00 – 16:00 | `hot_day` or `critical` (random) — potential critical alert |
| 16:00 – 20:00 | `hot_day` — cooling down |
| 20:00 – 22:00 | `normal` |
| any time (~8%) | `resource_low` — feeder/waterer low |

### Disable (switch to live Pi mode)

```
SIMULATION_MODE=false   # or delete the line entirely
```

Restart uvicorn — nothing else changes. The Pi data will appear as normal.

---

## Troubleshooting

### Error: "Cannot connect to database"

**Fix:**
1. Check PostgreSQL is running:
   ```bash
   # Windows (adjust version number to match your install, e.g. 16, 17, 18)
   net start postgresql-x64-18

   # Mac
   brew services list | grep postgresql

   # Linux
   sudo service postgresql status
   ```

2. Test connection manually:
   ```bash
   psql -d chickens
   # If this fails, your password is wrong in .env
   ```

3. Check `.env` has correct DB_PASSWORD

### Error: "Ollama model not found"

**Fix:**
```bash
ollama pull qwen2.5:1.5b-instruct
ollama pull nomic-embed-text
ollama list  # Verify they're there
```

### Error: "No such file or directory: test_docs"

**Fix:**
```bash
mkdir test_docs
echo "Test content" > test_docs/test.txt
```

### Error: "ModuleNotFoundError: No module named 'langchain'"

**Fix:**
```bash
# Make sure venv is activated (you should see (venv) in terminal)
pip install -r requirements.txt
```

### Slow performance (>10s per query)

**Try:**
1. Use smaller model: change `OLLAMA_MODEL` in `.env`
2. Reduce chunks in `rag_functions.py`: `chunk_size=400`, `k=2`
3. Delete `chroma_db/` and rebuild

### Database already exists error

**Fix:**
```bash
# Drop and recreate
dropdb chickens
createdb chickens
python generate_demo_data.py
```

---

## Expected File Structure After Setup

```
your-project/
├── venv/                          # Python virtual environment (gitignored)
├── chroma_db/                     # Vector database (auto-created, gitignored)
├── test_docs/                     # Your knowledge base
│   └── chicken_basics.txt
├── frontend/                      # React web UI
│   ├── node_modules/              # JS deps (gitignored — run npm install)
│   ├── dist/                      # Production build (gitignored — run npm run build)
│   ├── src/
│   │   ├── components/
│   │   │   ├── ChatPanel.jsx
│   │   │   ├── SensorDashboard.jsx
│   │   │   ├── SensorChart.jsx
│   │   │   ├── AlertFeed.jsx
│   │   │   └── Layout.jsx
│   │   ├── api/index.js
│   │   ├── App.jsx
│   │   └── main.jsx
│   ├── index.html
│   ├── package.json
│   ├── vite.config.js
│   ├── tailwind.config.js
│   └── postcss.config.js
├── evaluation/                    # RAG evaluation scripts
│   ├── evaluate_rag.py            # RAG vs NO-RAG heuristic comparison
│   ├── evaluate_ragas.py          # RAGAS semantic evaluation
│   ├── evaluate_deepeval.py       # DeepEval G-Eval metrics
│   ├── evaluate_retrieval.py      # Retrieval method comparison
│   ├── evaluation_data.py         # Test questions + ground truth
│   ├── eval_config.py             # Scoring criteria
│   └── TEAMMATE_README.md         # Instructions for evaluation content task
├── .env                           # Database + model config (gitignored — don't commit!)
├── .env.example                   # Safe template to share with teammates
├── db_utils.py                    # Database queries (sensor_readings + event_log)
├── sensor_filter.py               # Decide when to include sensor data in prompts
├── prompts.py                     # System prompts for LLM
├── rag_functions.py               # Complete RAG pipeline (hybrid retrieval)
├── pi_sensor_writer.py            # Template for Pi sensor team to write data
├── generate_demo_data.py          # Create test sensor readings
├── scheduler.py                   # Background sensor monitoring + alerts
├── app.py                         # FastAPI server + serves frontend/dist
├── requirements.txt               # Python dependencies
├── README.md                      # Project overview and documentation
└── COMPLETE_SETUP_GUIDE.md        # This file
```

---

## What Each File Does

| File | Purpose |
|------|---------|
| `db_utils.py` | Connect to PostgreSQL, query sensors, log events |
| `sensor_filter.py` | Decide when to include sensor data in LLM prompts |
| `prompts.py` | System prompts for LLM |
| `rag_functions.py` | Complete RAG pipeline (hybrid retrieval) |
| `pi_sensor_writer.py` | Template for sensor team to push data from Pi |
| `generate_demo_data.py` | Create test sensor readings |
| `scheduler.py` | Background: checks sensors, triggers RAG on critical, logs to event_log |
| `app.py` | FastAPI server with /ask, /sensors, /events endpoints |
| `evaluation/evaluate_rag.py` | Compare RAG vs NO-RAG performance |
| `evaluation/evaluate_ragas.py` | RAGAS semantic evaluation |
| `evaluation/evaluate_deepeval.py` | DeepEval G-Eval custom metrics |
| `evaluation/evaluate_retrieval.py` | Retrieval method comparison |
| `evaluation/evaluation_data.py` | Test questions + ground truth |
| `evaluation/eval_config.py` | Scoring criteria for G-Eval |

---

## Checklist

- [ ] PostgreSQL installed and running (`net start postgresql-x64-18` on Windows)
- [ ] Database `chickens` created
- [ ] Ollama installed with both models (`qwen2.5:1.5b-instruct` + `nomic-embed-text`)
- [ ] Node.js 18+ installed (`node --version` to verify)
- [ ] Python venv created and activated
- [ ] Python dependencies installed (`pip install -r requirements.txt`)
- [ ] Frontend dependencies installed (`cd frontend && npm install`)
- [ ] `.env` configured with DB credentials
- [ ] `test_docs/` folder created with at least 1 file
- [ ] Demo data generated successfully (`python generate_demo_data.py`)
- [ ] Frontend built (`cd frontend && npm run build`)
- [ ] RAG pipeline runs without errors (`python rag_functions.py`)
- [ ] Server starts and UI loads at http://localhost:8000
- [ ] Evaluation completes successfully

---

**You're ready!**

Start with: `python generate_demo_data.py`
