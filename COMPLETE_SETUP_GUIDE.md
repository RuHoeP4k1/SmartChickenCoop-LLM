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

## Step 4: Setup Python Environment

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

## Step 5: Install Python Dependencies

```bash
# Make sure venv is activated!
pip install -r requirements.txt

# This will install:
# - psycopg2-binary (PostgreSQL)
# - langchain + chromadb (RAG)
# - pypdf (document loading)
# - fastapi (API server)

# Should take 2-3 minutes
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

## Step 9: Test the RAG Pipeline

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

## Step 10: Run Evaluation (RAG vs NO-RAG)

```bash
python evaluation/evaluate_rag.py

# This takes 5-10 minutes
# Tests 10 questions comparing RAG vs NO-RAG
```

---

## Step 11: Test Pi Sensor Writer

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
# Generate new demo data
python generate_demo_data.py

# Test RAG pipeline
python rag_functions.py

# Run evaluation
python evaluation/evaluate_rag.py

# Test Pi sensor writer
python pi_sensor_writer.py

# Check latest sensor reading
python -c "from db_utils import get_latest_sensor_reading; print(get_latest_sensor_reading())"

# Rebuild vector database (if you add new documents)
rm -rf chroma_db/  # Windows: rmdir /s chroma_db
python rag_functions.py

# Start the API server
uvicorn app:app --reload
```

---

## Troubleshooting

### Error: "Cannot connect to database"

**Fix:**
1. Check PostgreSQL is running:
   ```bash
   # Windows
   net start postgresql-x64-16

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
├── app.py                         # FastAPI server
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

- [ ] PostgreSQL installed and running
- [ ] Database `chickens` created
- [ ] Ollama installed with both models
- [ ] Python venv created and activated
- [ ] Dependencies installed (`pip install -r requirements.txt`)
- [ ] `.env` configured with DB credentials
- [ ] `test_docs/` folder created with at least 1 file
- [ ] Demo data generated successfully
- [ ] RAG pipeline runs without errors
- [ ] Evaluation completes successfully

---

**You're ready!**

Start with: `python generate_demo_data.py`
