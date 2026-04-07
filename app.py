"""
ChickenCare AI - FastAPI Application
Run with: uvicorn app:app --reload
"""

import os
import asyncio
import calendar
import logging
import secrets
import httpx
from contextlib import asynccontextmanager
from pathlib import Path
from datetime import datetime, date, time as dtime, timedelta
from fastapi import FastAPI, HTTPException, Query, UploadFile, File, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator, StringConstraints
from typing import Optional, List, Annotated

# Bounded string type for list elements (prevents unbounded JSONB blobs)
BoundedStr = Annotated[str, StringConstraints(max_length=200)]

from backend.rag_functions import (
    load_documents, split_documents, build_vector_store,
    build_bm25_retriever, answer_query
)
from backend.db_utils import (
    get_latest_sensor_reading, setup_database,
    get_recent_readings, get_recent_events, insert_event,
    get_sensor_history, upsert_review, get_events_for_review,
    export_reviews,
    upsert_egg_entry, get_egg_entries_for_month,
    get_chore_definitions, insert_chore_log, delete_chore_log,
    get_chore_log_in_range,
    insert_automation_window, delete_automation_window,
    get_automation_windows_in_range,
    ALLOWED_AUTOMATION_TASKS,
)
from backend.sensor_filter import get_sensor_context
from backend.scheduler import start_scheduler, stop_scheduler
from backend.egg_reconciliation import reconcile_day
from backend.consumption import consumption_for_date

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

KNOWLEDGE_BASE_PATH  = os.getenv("KNOWLEDGE_BASE_PATH", "test_docs")
SIMULATION_MODE      = os.getenv("SIMULATION_MODE", "false").lower() == "true"
SCHEDULER_INTERVAL   = int(os.getenv("SCHEDULER_INTERVAL", "60" if SIMULATION_MODE else "600"))
CHAT_HISTORY_TURNS   = int(os.getenv("CHAT_HISTORY_TURNS", "2"))
COOP_LAT             = float(os.getenv("COOP_LAT", "50.8798"))
COOP_LON             = float(os.getenv("COOP_LON", "4.7005"))


async def _build_rag_pipeline():
    """Build RAG pipeline in the background so startup doesn't block the server."""
    if not os.path.exists(KNOWLEDGE_BASE_PATH):
        logger.warning(f"Knowledge base path '{KNOWLEDGE_BASE_PATH}' not found.")
        return
    docs = load_documents(KNOWLEDGE_BASE_PATH)
    chunks = split_documents(docs)
    state.vectordb = await asyncio.to_thread(lambda: build_vector_store(chunks, folder_path=KNOWLEDGE_BASE_PATH))
    state.bm25_retriever = build_bm25_retriever(chunks)
    state.ready = True
    logger.info(f"RAG pipeline ready ({len(chunks)} chunks)")
    # Start scheduler once RAG is ready
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


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Start server immediately; build RAG pipeline in background."""
    logger.info("Starting ChickenCare AI...")

    # Ensure DB tables exist (safe to run every startup — uses IF NOT EXISTS)
    try:
        setup_database()
    except Exception as e:
        logger.warning(f"DB setup failed (is PostgreSQL running?): {e}")

    # Fire RAG build in background — server starts serving immediately
    asyncio.create_task(_build_rag_pipeline())

    yield

    stop_scheduler()
    logger.info("Shutting down ChickenCare AI.")


app = FastAPI(
    title="ChickenCare AI",
    description="AI-powered chicken welfare assistant with sensor integration",
    version="1.2.0",
    lifespan=lifespan,
)

_allowed_origins_env = os.getenv("ALLOWED_ORIGINS", "")
_allowed_origins = [o.strip() for o in _allowed_origins_env.split(",") if o.strip()] or ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=_allowed_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# API key auth

_API_KEY = os.getenv("API_KEY")
if not _API_KEY:
    logging.warning("API_KEY env var not set — POST /ask is unprotected")

_bearer = HTTPBearer(auto_error=False)


def verify_key(credentials: HTTPAuthorizationCredentials = Depends(_bearer)):
    if not _API_KEY:
        return  # no key configured → open for local dev
    if not credentials or not secrets.compare_digest(credentials.credentials, _API_KEY):
        raise HTTPException(status_code=401, detail="Invalid or missing API key")


# ---------------------------------------------------------------------------
# Request/response models
# ---------------------------------------------------------------------------

class QueryRequest(BaseModel):
    query: str
    use_sensors: bool = True
    use_hybrid: bool = True
    include_task_context: bool = False
    history: list[dict] = []


class QueryResponse(BaseModel):
    query: str
    answer: str
    sources: list[str]
    chunks: list[dict]
    sensor_included: bool
    sensor_context: Optional[str]
    has_critical: bool
    retrieval_method: str


class ReviewRequest(BaseModel):
    event_id: int
    is_good: bool
    routing_correct: Optional[bool] = None
    notes: str = ""



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
def ask_question(request: QueryRequest, _=Depends(verify_key)):
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

    # Trim history to the configured number of turns (each turn = 1 user + 1 assistant msg)
    trimmed_history = request.history[-(CHAT_HISTORY_TURNS * 2):] if CHAT_HISTORY_TURNS > 0 else []

    result = answer_query(
        query=request.query,
        vectordb=state.vectordb,
        bm25_retriever=state.bm25_retriever,
        use_sensors=request.use_sensors,
        use_hybrid=request.use_hybrid,
        history=trimmed_history,
        include_task_context=request.include_task_context,
    )

    # Log to event_log for later evaluation
    try:
        insert_event(
            event_type="llm_response",
            severity="critical" if result["has_critical"] else "info",
            user_query=request.query,
            llm_response=result["answer"],
            sensor_snapshot=result["sensor_data"],
            sensor_context_filtered=result["sensor_context"],
            sources=result["sources"],
            routing_mode=result.get("routing_mode"),
            routing_decision=result.get("routing_decision"),
            prompt_template=result.get("prompt_template"),
            response_time_ms=result.get("response_time_ms"),
        )
    except Exception as e:
        logger.warning(f"Failed to log event: {e}")

    chunks = [
        {"source": doc.metadata.get("source", "Unknown"), "content": doc.page_content}
        for doc in result.get("documents", [])
    ]
    return QueryResponse(**{
        **{k: v for k, v in result.items() if k not in ("documents", "sensor_data")},
        "chunks": chunks,
    })


@app.post("/setup-db")
def setup_db():
    """Create database tables (run once)."""
    try:
        setup_database()
        return {"status": "ok", "message": "Database tables created (sensor_readings_colson + cv_counts_colson + event_log)"}
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


@app.get("/weather")
def get_weather():
    """
    Fetch current weather + 3-day forecast for the coop location via Open-Meteo.
    No API key required. Coordinates set via COOP_LAT / COOP_LON in .env.
    """
    url = (
        "https://api.open-meteo.com/v1/forecast"
        f"?latitude={COOP_LAT}&longitude={COOP_LON}"
        "&current=temperature_2m,relative_humidity_2m,apparent_temperature,"
        "precipitation,weather_code,wind_speed_10m"
        "&daily=temperature_2m_max,temperature_2m_min,precipitation_sum,"
        "sunrise,sunset,daylight_duration,weather_code"
        "&timezone=Europe%2FBrussels&forecast_days=3"
    )
    try:
        resp = httpx.get(url, timeout=8)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Weather API error: {e}")


# ---------------------------------------------------------------------------
# Heatmap endpoints — CV pipeline on Pi pushes images here
# ---------------------------------------------------------------------------

_HEATMAP_DIR = Path("uploads/heatmaps")
_HEATMAP_DIR.mkdir(parents=True, exist_ok=True)


@app.get("/heatmap/latest")
async def get_latest_heatmap():
    """Return the most recently uploaded heatmap image metadata, or 404 if none."""
    images = list(_HEATMAP_DIR.glob("*.png")) + list(_HEATMAP_DIR.glob("*.jpg"))
    if not images:
        raise HTTPException(status_code=404, detail="No heatmap available yet")
    latest = max(images, key=lambda f: f.stat().st_mtime)
    return {
        "url": f"/static/uploads/heatmaps/{latest.name}",
        "filename": latest.name,
        "timestamp": latest.stat().st_mtime,
    }


@app.post("/heatmap/upload")
async def upload_heatmap(file: UploadFile = File(...)):
    """Accept a heatmap image from the CV pipeline (Raspberry Pi)."""
    ts = int(datetime.utcnow().timestamp())
    filename = f"{ts}_{file.filename}"
    dest = _HEATMAP_DIR / filename
    dest.write_bytes(await file.read())
    logger.info(f"Heatmap received: {filename}")
    return {"url": f"/static/uploads/heatmaps/{filename}", "filename": filename}


# ---------------------------------------------------------------------------
# Review endpoints (for live response evaluation)
# ---------------------------------------------------------------------------

@app.get("/reviews")
def get_reviews(limit: int = 50, reviewed: Optional[str] = None):
    """Get llm_response events with review status for the Review tab."""
    reviewed_bool = None
    if reviewed == "true":
        reviewed_bool = True
    elif reviewed == "false":
        reviewed_bool = False
    events = get_events_for_review(limit=min(limit, 200), reviewed=reviewed_bool)
    return {"events": events, "count": len(events)}


@app.post("/reviews")
def submit_review(review: ReviewRequest):
    """Submit or update a review for an event_log entry."""
    review_id = upsert_review(
        event_id=review.event_id,
        is_good=review.is_good,
        routing_correct=review.routing_correct,
        notes=review.notes,
    )
    return {"id": review_id, "status": "ok"}


@app.get("/reviews/export")
def export_reviews_endpoint():
    """Export all events + reviews as JSON (frontend converts to CSV)."""
    rows = export_reviews()
    return {"data": rows, "count": len(rows)}


# ---------------------------------------------------------------------------
# Coop Log endpoints — eggs, chores, automation windows
# ---------------------------------------------------------------------------

class EggEntryRequest(BaseModel):
    eggs_laid: int = Field(ge=0, le=100)


class ChoreLogRequest(BaseModel):
    entry_date: date
    definition_id: int = Field(ge=1)
    checked_items: List[BoundedStr] = Field(default_factory=list, max_length=20)
    notes: str = Field(default="", max_length=500)


class AutomationWindowRequest(BaseModel):
    task: str
    start_date: date
    end_date: date
    start_time: Optional[dtime] = None
    end_time: Optional[dtime] = None
    days_of_week: List[int] = Field(default_factory=lambda: [0, 1, 2, 3, 4, 5, 6])

    @field_validator("task")
    @classmethod
    def _task_allowed(cls, v: str) -> str:
        if v not in ALLOWED_AUTOMATION_TASKS:
            raise ValueError(f"task must be one of {ALLOWED_AUTOMATION_TASKS}")
        return v

    @field_validator("days_of_week")
    @classmethod
    def _dow_range(cls, v: List[int]) -> List[int]:
        for d in v:
            if d < 0 or d > 6:
                raise ValueError("days_of_week entries must be 0..6 (Sun..Sat)")
        return v


_MAX_AUTOMATION_SPAN_DAYS = 366


def _db_error(e: Exception, label: str = "Database error") -> HTTPException:
    """Log the full exception server-side, return a generic 503 to the client."""
    logger.warning(f"{label}: {e}")
    return HTTPException(status_code=503, detail=label)


@app.get("/eggs/calendar")
def get_coop_log_month(
    year: int = Query(..., ge=2000, le=2100, description="4-digit year"),
    month: int = Query(..., ge=1, le=12),
    include_consumption: bool = Query(True),
):
    """
    Return a combined month view for the Coop Log tab:
    - egg entries (auto-reconciled + manual)
    - chores logged that month
    - automation windows overlapping that month
    - per-day rollup: eggs, chores count, automation hours, feed/water % consumed
    """
    days_in_month = calendar.monthrange(year, month)[1]
    first = date(year, month, 1)
    last = date(year, month, days_in_month)

    try:
        eggs = get_egg_entries_for_month(year, month)
        chores = get_chore_log_in_range(first, last)
        windows = get_automation_windows_in_range(first, last)
    except Exception as e:
        raise _db_error(e)

    # Per-day rollup
    eggs_by_day = {e["entry_date"].isoformat(): e for e in eggs}
    chores_by_day: dict[str, list] = {}
    for c in chores:
        key = c["entry_date"].isoformat()
        chores_by_day.setdefault(key, []).append(c)

    today = date.today()
    rollup = []
    for day_num in range(1, days_in_month + 1):
        d = date(year, month, day_num)
        key = d.isoformat()

        egg_entry = eggs_by_day.get(key)
        day_chores = chores_by_day.get(key, [])

        # Automation hours active that day
        auto_hours = 0.0
        active_tasks = []
        for w in windows:
            if w["start_date"] > d or w["end_date"] < d:
                continue
            # day-of-week filter (Sun=0..Sat=6 matches Python's .weekday() + 1 % 7)
            py_dow = (d.weekday() + 1) % 7
            if py_dow not in (w.get("days_of_week") or []):
                continue
            st = w.get("start_time")
            et = w.get("end_time")
            if st is None or et is None:
                hours = 24.0
            else:
                hours = ((datetime.combine(d, et) - datetime.combine(d, st)).total_seconds()) / 3600.0
                if hours < 0:
                    hours += 24.0
            auto_hours += hours
            active_tasks.append(w["task"])

        consumption = {"feeder_pct_consumed": None, "waterer_pct_consumed": None}
        # Only compute consumption for past days and current day (future = skip)
        if include_consumption and d <= today:
            try:
                consumption = consumption_for_date(d)
            except Exception as e:
                logger.warning(f"consumption calc failed for {d}: {e}")

        rollup.append({
            "date": key,
            "eggs_laid": egg_entry["eggs_laid"] if egg_entry else 0,
            "eggs_source": egg_entry["source"] if egg_entry else None,
            "chore_count": len(day_chores),
            "chore_labels": [c["label"] for c in day_chores],
            "automation_hours": round(auto_hours, 1),
            "automation_tasks": sorted(set(active_tasks)),
            "feeder_pct_consumed": consumption["feeder_pct_consumed"],
            "waterer_pct_consumed": consumption["waterer_pct_consumed"],
        })

    return {
        "year": year,
        "month": month,
        "days": rollup,
        "chores": chores,
        "automation_windows": windows,
    }


@app.post("/eggs/calendar/{entry_date}")
def set_egg_entry(entry_date: date, body: EggEntryRequest):
    """Manually set (or correct) the egg count for a specific date."""
    try:
        upsert_egg_entry(entry_date, body.eggs_laid, source="manual")
    except Exception as e:
        raise _db_error(e)
    return {"status": "ok", "entry_date": entry_date.isoformat(), "eggs_laid": body.eggs_laid}


@app.post("/eggs/reconcile")
def trigger_reconcile(entry_date: Optional[date] = Query(None)):
    """Manually trigger egg reconciliation for a date (default: yesterday)."""
    try:
        eggs = reconcile_day(entry_date)
    except Exception as e:
        raise _db_error(e, "Reconciliation failed")
    return {
        "status": "ok",
        "eggs_laid": eggs,
        "entry_date": (entry_date or (date.today() - timedelta(days=1))).isoformat(),
    }


@app.get("/chores/definitions")
def list_chore_definitions():
    """Return all active chore definitions for the 'Add chore' picker."""
    try:
        return {"definitions": get_chore_definitions()}
    except Exception as e:
        raise _db_error(e)


@app.post("/chores/log")
def add_chore_log(body: ChoreLogRequest):
    """Record a completed chore for a specific date."""
    try:
        log_id = insert_chore_log(
            entry_date=body.entry_date,
            definition_id=body.definition_id,
            checked_items=body.checked_items,
            notes=body.notes,
        )
    except Exception as e:
        raise _db_error(e)
    return {"status": "ok", "id": log_id}


@app.delete("/chores/log/{log_id}")
def remove_chore_log(log_id: int):
    try:
        removed = delete_chore_log(log_id)
    except Exception as e:
        raise _db_error(e)
    if removed == 0:
        raise HTTPException(status_code=404, detail="chore log entry not found")
    return {"status": "ok"}


@app.get("/automation/windows")
def list_automation_windows(
    start: date = Query(...),
    end: date = Query(...),
):
    if end < start:
        raise HTTPException(status_code=400, detail="end must be >= start")
    if (end - start).days > _MAX_AUTOMATION_SPAN_DAYS:
        raise HTTPException(
            status_code=400,
            detail=f"date range too large (max {_MAX_AUTOMATION_SPAN_DAYS} days)",
        )
    try:
        return {"windows": get_automation_windows_in_range(start, end)}
    except Exception as e:
        raise _db_error(e)


@app.post("/automation/windows")
def create_automation_window(body: AutomationWindowRequest):
    if body.end_date < body.start_date:
        raise HTTPException(status_code=400, detail="end_date must be >= start_date")
    if (body.end_date - body.start_date).days > _MAX_AUTOMATION_SPAN_DAYS:
        raise HTTPException(
            status_code=400,
            detail=f"window too long (max {_MAX_AUTOMATION_SPAN_DAYS} days)",
        )
    if (body.start_time is None) != (body.end_time is None):
        raise HTTPException(
            status_code=400,
            detail="start_time and end_time must both be set or both omitted",
        )
    try:
        win_id = insert_automation_window(
            task=body.task,
            start_date=body.start_date,
            end_date=body.end_date,
            start_time=body.start_time,
            end_time=body.end_time,
            days_of_week=body.days_of_week,
        )
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise _db_error(e)
    return {"status": "ok", "id": win_id}


@app.delete("/automation/windows/{win_id}")
def remove_automation_window(win_id: int):
    try:
        removed = delete_automation_window(win_id)
    except Exception as e:
        raise _db_error(e)
    if removed == 0:
        raise HTTPException(status_code=404, detail="automation window not found")
    return {"status": "ok"}


# Serve uploaded heatmap images as static files
app.mount("/static/uploads", StaticFiles(directory="uploads"), name="uploads")


# ---------------------------------------------------------------------------
# Serve the React frontend (must be LAST — after all API routes)
# Only active once you've run `npm run build` inside frontend/
# ---------------------------------------------------------------------------

if os.path.exists("frontend/dist"):
    app.mount("/", StaticFiles(directory="frontend/dist", html=True), name="frontend")
