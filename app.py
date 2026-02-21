"""
ChickenCare AI - FastAPI Application
Run with: uvicorn app:app --reload
"""

import os
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Optional

from rag_functions import (
    load_documents, split_documents, build_vector_store,
    build_bm25_retriever, answer_query
)
from db_utils import (
    get_latest_sensor_reading, setup_database,
    get_recent_readings, get_recent_events, insert_event,
    get_sensor_history,
)
from sensor_filter import get_sensor_context
from scheduler import start_scheduler, stop_scheduler

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("chickencareai")


# ---------------------------------------------------------------------------
# App state — initialized once at startup
# ---------------------------------------------------------------------------

class AppState:
    vectordb = None
    bm25_retriever = None
    ready = False


state = AppState()

KNOWLEDGE_BASE_PATH = os.getenv("KNOWLEDGE_BASE_PATH", "test_docs")
SCHEDULER_INTERVAL  = int(os.getenv("SCHEDULER_INTERVAL", "60"))   # seconds
SIMULATION_MODE     = os.getenv("SIMULATION_MODE", "false").lower() == "true"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load knowledge base, build retrievers, start scheduler on startup."""
    logger.info("Starting ChickenCare AI...")

    # Build RAG pipeline once
    if os.path.exists(KNOWLEDGE_BASE_PATH):
        docs = load_documents(KNOWLEDGE_BASE_PATH)
        chunks = split_documents(docs)
        state.vectordb = build_vector_store(chunks, folder_path=KNOWLEDGE_BASE_PATH)
        state.bm25_retriever = build_bm25_retriever(chunks)
        state.ready = True
        logger.info(f"RAG pipeline ready ({len(chunks)} chunks)")
    else:
        logger.warning(f"Knowledge base path '{KNOWLEDGE_BASE_PATH}' not found.")

    # Start background sensor monitoring
    try:
        start_scheduler(
            interval_seconds=SCHEDULER_INTERVAL,
            vectordb=state.vectordb,
            bm25_retriever=state.bm25_retriever,
            simulation_mode=SIMULATION_MODE,
        )
        if SIMULATION_MODE:
            logger.info("SIMULATION MODE ON — fake readings every 60 s, scheduler every %ds", SCHEDULER_INTERVAL)
    except Exception as e:
        logger.warning(f"Scheduler failed to start (DB might not be ready): {e}")

    yield

    stop_scheduler()
    logger.info("Shutting down ChickenCare AI.")


app = FastAPI(
    title="ChickenCare AI",
    description="AI-powered chicken welfare assistant with sensor integration",
    version="1.2.0",
    lifespan=lifespan,
)

# Allow frontend on any origin during development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Request/response models
# ---------------------------------------------------------------------------

class QueryRequest(BaseModel):
    query: str
    use_sensors: bool = True
    use_hybrid: bool = True


class QueryResponse(BaseModel):
    query: str
    answer: str
    sources: list[str]
    sensor_included: bool
    sensor_context: Optional[str]
    has_critical: bool
    retrieval_method: str



# ---------------------------------------------------------------------------
# Core endpoints
# ---------------------------------------------------------------------------

@app.get("/api/info")
def root():
    """API info endpoint (moved from / to avoid conflicting with the React frontend)."""
    return {
        "app": "ChickenCare AI",
        "version": "1.2.0",
        "rag_ready": state.ready,
        "docs": "/docs",
        "endpoints": {
            "core":    ["/health", "/ask", "/setup-db"],
            "sensors": ["/sensors"],
            "events":  ["/events"],
        },
    }


@app.get("/health")
def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "rag_ready": state.ready,
    }


@app.post("/ask", response_model=QueryResponse)
def ask_question(request: QueryRequest):
    """
    Ask a chicken-keeping question.
    Uses RAG pipeline with optional sensor context.
    Logs the query + response to event_log for evaluation.
    """
    if not state.ready:
        raise HTTPException(
            status_code=503,
            detail="RAG pipeline not initialized. Add documents to the knowledge base.",
        )

    result = answer_query(
        query=request.query,
        vectordb=state.vectordb,
        bm25_retriever=state.bm25_retriever,
        use_sensors=request.use_sensors,
        use_hybrid=request.use_hybrid,
    )

    # Log to event_log for later evaluation
    try:
        sensor_data = get_latest_sensor_reading() if request.use_sensors else None
        insert_event(
            event_type="llm_response",
            severity="critical" if result["has_critical"] else "info",
            user_query=request.query,
            llm_response=result["answer"],
            sensor_snapshot=sensor_data,
            sensor_context_filtered=result["sensor_context"],
        )
    except Exception as e:
        logger.warning(f"Failed to log event: {e}")

    return QueryResponse(**result)


@app.post("/setup-db")
def setup_db():
    """Create database tables (run once)."""
    try:
        setup_database()
        return {"status": "ok", "message": "Database tables created (sensor_readings + event_log)"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Setup failed: {e}")


# ---------------------------------------------------------------------------
# Sensor endpoints
# ---------------------------------------------------------------------------

@app.get("/sensors")
def get_sensors():
    """Get the latest sensor reading."""
    try:
        reading = get_latest_sensor_reading()
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Database error: {e}")

    if not reading:
        raise HTTPException(status_code=404, detail="No sensor data available")

    return {
        "reading": reading,
        "summary": get_sensor_context(reading),
    }


# ---------------------------------------------------------------------------
# Event log endpoints — for reviewing system performance + scheduler alerts
# ---------------------------------------------------------------------------

@app.get("/events")
def get_events(
    limit: int = Query(20, ge=1, le=100),
    event_type: Optional[str] = Query(None, description="Filter: llm_response, sensor_alert, conditions_normal"),
):
    """
    Get recent events from the event log.
    The scheduler writes sensor_alert / conditions_normal events here.
    The /ask endpoint writes llm_response events here.
    Use this to review system performance or show alerts in the frontend.
    """
    try:
        events = get_recent_events(limit=limit, event_type=event_type)
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Database error: {e}")

    return {"events": events, "count": len(events)}


# ---------------------------------------------------------------------------
# Sensor history endpoint — for the frontend chart
# ---------------------------------------------------------------------------

@app.get("/sensors/history")
def get_history(
    range: str = Query("1h", description="Time range: 1h, 24h, 7d"),
):
    """
    Return sensor readings for the past 1h / 24h / 7d.
    Used by the frontend chart to plot historical trends.
    """
    range_hours = {"1h": 1, "24h": 24, "7d": 168}
    range_limit = {"1h": 120, "24h": 300, "7d": 500}

    hours = range_hours.get(range, 1)
    limit = range_limit.get(range, 120)

    try:
        readings = get_sensor_history(hours=hours, limit=limit)
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Database error: {e}")

    return {"readings": readings, "range": range, "count": len(readings)}


# ---------------------------------------------------------------------------
# Serve the React frontend (must be LAST — after all API routes)
# Only active once you've run `npm run build` inside frontend/
# ---------------------------------------------------------------------------

if os.path.exists("frontend/dist"):
    app.mount("/", StaticFiles(directory="frontend/dist", html=True), name="frontend")
