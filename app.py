"""
ChickenCare AI - FastAPI Application
Run with: uvicorn app:app --reload
"""

import os
import asyncio
import calendar
import logging
import httpx
from contextlib import asynccontextmanager
from pathlib import Path
from datetime import datetime, date, time as dtime, timedelta
from fastapi import FastAPI, HTTPException, Query, UploadFile, File, Depends
from fastapi.security import OAuth2PasswordBearer
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field, field_validator, StringConstraints
from typing import Optional, List, Annotated
from jose import JWTError
from backend.auth import hash_password, verify_password, create_access_token, decode_token
from backend.db_utils import create_user, get_user_by_username

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
    get_latest_risk_snapshot,
)
from backend.sensor_filter import (
    get_sensor_context, get_critical_alerts,
    heat_is_non_normal, mold_is_non_normal,
)
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
# JWT auth

_oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/login")


def get_current_user(token: str = Depends(_oauth2_scheme)) -> dict:
    """Decode JWT and return {user_id, username}. Raises 401 on invalid/expired token."""
    try:
        payload = decode_token(token)
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid or expired token")
    return {"user_id": int(payload["sub"]), "username": payload["username"]}


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


class RegisterRequest(BaseModel):
    username: Annotated[str, StringConstraints(min_length=3, max_length=50)]
    password: Annotated[str, StringConstraints(min_length=8, max_length=100)]


class LoginRequest(BaseModel):
    username: str
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str
    username: str
    user_id: int


# ---------------------------------------------------------------------------
# Auth endpoints
# ---------------------------------------------------------------------------

@app.post("/auth/register", status_code=201)
def register(body: RegisterRequest):
    """Create a new user account."""
    if get_user_by_username(body.username):
        raise HTTPException(status_code=409, detail="Username already taken")
    hashed = hash_password(body.password)
    user_id = create_user(body.username, hashed)
    return {"id": user_id, "username": body.username}


@app.post("/auth/login", response_model=TokenResponse)
def login(body: LoginRequest):
    """Authenticate and return a JWT access token."""
    user = get_user_by_username(body.username)
    if not user or not verify_password(body.password, user["hashed_pw"]):
        raise HTTPException(status_code=401, detail="Invalid username or password")
    token = create_access_token(user["id"], user["username"])
    return TokenResponse(
        access_token=token,
        token_type="bearer",
        username=user["username"],
        user_id=user["id"],
    )


@app.get("/auth/me")
def me(current_user: dict = Depends(get_current_user)):
    """Return the currently authenticated user."""
    return {"user_id": current_user["user_id"], "username": current_user["username"]}


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
def ask_question(request: QueryRequest, current_user: dict = Depends(get_current_user)):
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
            owner_id=current_user["user_id"],
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
def get_sensors(current_user: dict = Depends(get_current_user)):
    """Get the latest sensor reading."""
    try:
        reading = get_latest_sensor_reading(owner_id=current_user["user_id"])
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
    current_user: dict = Depends(get_current_user),
):
    """Get recent events — user's own events plus system events (owner_id IS NULL)."""
    try:
        events = get_recent_events(limit=limit, event_type=event_type, owner_id=current_user["user_id"])
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Database error: {e}")

    return {"events": events, "count": len(events)}


# ---------------------------------------------------------------------------
# Sensor history endpoint — for the frontend chart
# ---------------------------------------------------------------------------

@app.get("/sensors/history")
def get_history(
    range: str = Query("1h", description="Time range: 1h, 24h, 7d"),
    current_user: dict = Depends(get_current_user),
):
    """Return sensor readings for the past 1h / 24h / 7d."""
    range_hours = {"1h": 1, "24h": 24, "7d": 168}
    range_limit = {"1h": 120, "24h": 300, "7d": 500}

    hours = range_hours.get(range, 1)
    limit = range_limit.get(range, 120)

    try:
        readings = get_sensor_history(hours=hours, limit=limit, owner_id=current_user["user_id"])
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Database error: {e}")

    return {"readings": readings, "range": range, "count": len(readings)}


@app.get("/risk/latest")
def get_risk_latest(current_user: dict = Depends(get_current_user)):
    """Return the most recent risk snapshot (heat + mold + ventilation metrics)."""
    try:
        snapshot = get_latest_risk_snapshot(owner_id=current_user["user_id"])
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Database error: {e}")

    if not snapshot:
        raise HTTPException(status_code=404, detail="No risk snapshot available")

    return {"snapshot": snapshot}


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
async def get_latest_heatmap(current_user: dict = Depends(get_current_user)):
    """Return the most recently uploaded heatmap for the current user."""
    owner_dir = _HEATMAP_DIR / str(current_user["user_id"])
    owner_dir.mkdir(parents=True, exist_ok=True)
    images = list(owner_dir.glob("*.png")) + list(owner_dir.glob("*.jpg"))
    if not images:
        raise HTTPException(status_code=404, detail="No heatmap available yet")
    latest = max(images, key=lambda f: f.stat().st_mtime)
    return FileResponse(latest, media_type="image/png")


@app.post("/heatmap/upload")
async def upload_heatmap(file: UploadFile = File(...), current_user: dict = Depends(get_current_user)):
    """Accept a heatmap image from the CV pipeline (Raspberry Pi)."""
    owner_dir = _HEATMAP_DIR / str(current_user["user_id"])
    owner_dir.mkdir(parents=True, exist_ok=True)
    ts = int(datetime.utcnow().timestamp())
    filename = f"{ts}_{file.filename}"
    dest = owner_dir / filename
    dest.write_bytes(await file.read())
    logger.info(f"Heatmap received for user {current_user['user_id']}: {filename}")
    return {"filename": filename}


# ---------------------------------------------------------------------------
# Review endpoints (for live response evaluation)
# ---------------------------------------------------------------------------

@app.get("/reviews")
def get_reviews(limit: int = 50, reviewed: Optional[str] = None, current_user: dict = Depends(get_current_user)):
    """Get llm_response events with review status for the Review tab."""
    reviewed_bool = None
    if reviewed == "true":
        reviewed_bool = True
    elif reviewed == "false":
        reviewed_bool = False
    events = get_events_for_review(limit=min(limit, 200), reviewed=reviewed_bool, owner_id=current_user["user_id"])
    return {"events": events, "count": len(events)}


@app.post("/reviews")
def submit_review(review: ReviewRequest, current_user: dict = Depends(get_current_user)):
    """Submit or update a review for an event_log entry."""
    review_id = upsert_review(
        event_id=review.event_id,
        is_good=review.is_good,
        routing_correct=review.routing_correct,
        notes=review.notes,
    )
    return {"id": review_id, "status": "ok"}


@app.get("/reviews/export")
def export_reviews_endpoint(current_user: dict = Depends(get_current_user)):
    """Export all events + reviews as JSON (frontend converts to CSV)."""
    rows = export_reviews(owner_id=current_user["user_id"])
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


class AutomationEvaluateRequest(BaseModel):
    vent_enabled: bool = False
    door_enabled: bool = False


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
    current_user: dict = Depends(get_current_user),
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

    owner_id = current_user["user_id"]
    try:
        eggs = get_egg_entries_for_month(year, month, owner_id=owner_id)
        chores = get_chore_log_in_range(first, last, owner_id=owner_id)
        windows = get_automation_windows_in_range(first, last, owner_id=owner_id)
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
def set_egg_entry(entry_date: date, body: EggEntryRequest, current_user: dict = Depends(get_current_user)):
    """Manually set (or correct) the egg count for a specific date."""
    try:
        upsert_egg_entry(entry_date, body.eggs_laid, source="manual", owner_id=current_user["user_id"])
    except Exception as e:
        raise _db_error(e)
    return {"status": "ok", "entry_date": entry_date.isoformat(), "eggs_laid": body.eggs_laid}


@app.post("/eggs/reconcile")
def trigger_reconcile(entry_date: Optional[date] = Query(None), current_user: dict = Depends(get_current_user)):
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
def add_chore_log(body: ChoreLogRequest, current_user: dict = Depends(get_current_user)):
    """Record a completed chore for a specific date."""
    try:
        log_id = insert_chore_log(
            entry_date=body.entry_date,
            definition_id=body.definition_id,
            checked_items=body.checked_items,
            notes=body.notes,
            owner_id=current_user["user_id"],
        )
    except Exception as e:
        raise _db_error(e)
    return {"status": "ok", "id": log_id}


@app.delete("/chores/log/{log_id}")
def remove_chore_log(log_id: int, current_user: dict = Depends(get_current_user)):
    try:
        removed = delete_chore_log(log_id, owner_id=current_user["user_id"])
    except Exception as e:
        raise _db_error(e)
    if removed == 0:
        raise HTTPException(status_code=404, detail="chore log entry not found")
    return {"status": "ok"}


@app.get("/automation/windows")
def list_automation_windows(
    start: date = Query(...),
    end: date = Query(...),
    current_user: dict = Depends(get_current_user),
):
    if end < start:
        raise HTTPException(status_code=400, detail="end must be >= start")
    if (end - start).days > _MAX_AUTOMATION_SPAN_DAYS:
        raise HTTPException(
            status_code=400,
            detail=f"date range too large (max {_MAX_AUTOMATION_SPAN_DAYS} days)",
        )
    try:
        return {"windows": get_automation_windows_in_range(start, end, owner_id=current_user["user_id"])}
    except Exception as e:
        raise _db_error(e)


@app.post("/automation/windows")
def create_automation_window(body: AutomationWindowRequest, current_user: dict = Depends(get_current_user)):
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
            owner_id=current_user["user_id"],
        )
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise _db_error(e)
    return {"status": "ok", "id": win_id}


@app.delete("/automation/windows/{win_id}")
def remove_automation_window(win_id: int, current_user: dict = Depends(get_current_user)):
    try:
        removed = delete_automation_window(win_id, owner_id=current_user["user_id"])
    except Exception as e:
        raise _db_error(e)
    if removed == 0:
        raise HTTPException(status_code=404, detail="automation window not found")
    return {"status": "ok"}


@app.post("/automation/evaluate")
def evaluate_automation(body: AutomationEvaluateRequest, current_user: dict = Depends(get_current_user)):
    """
    Evaluate current conditions and determine what automation would do.
    Logs a decision to event_log and returns the action + reason.
    For show: does not control hardware, but accurately reflects what the
    ventilation pipeline would decide based on live sensor data.
    """
    sensor = get_latest_sensor_reading(owner_id=current_user["user_id"])
    if not sensor:
        raise HTTPException(status_code=404, detail="No sensor data available")

    actions = []

    if body.vent_enabled:
        heat_level = sensor.get("heat_risk_level") or ""
        mold_level = sensor.get("mold_risk_level") or ""
        co2_level = sensor.get("co2_level") or "normal"
        nh3_level = sensor.get("nh3_level") or "normal"
        h2s_level = sensor.get("h2s_level") or "normal"

        triggers = []
        if heat_is_non_normal(heat_level):
            short = heat_level.split(" - ")[0] if " - " in heat_level else heat_level
            triggers.append(f"heat stress {short}")
        if mold_is_non_normal(mold_level):
            triggers.append(f"mold risk {mold_level}")
        if co2_level in ("warning", "critical"):
            ppm = sensor.get("co2_ppm")
            triggers.append(f"CO2 {co2_level}" + (f" ({ppm:.0f} ppm)" if ppm else ""))
        if nh3_level in ("warning", "critical"):
            ppm = sensor.get("nh3_ppm")
            triggers.append(f"NH3 {nh3_level}" + (f" ({ppm:.1f} ppm)" if ppm else ""))
        if h2s_level in ("warning", "critical"):
            ppm = sensor.get("h2s_ppm")
            triggers.append(f"H2S {h2s_level}" + (f" ({ppm:.1f} ppm)" if ppm else ""))

        if triggers:
            reason = "Ventilation boosted — " + ", ".join(triggers)
            risk_factors = sensor.get("risk_factors")
            if risk_factors:
                reason += f" | {risk_factors}"
            severity = "critical" if any(
                sensor.get(k) == "critical"
                for k in ("co2_level", "nh3_level", "h2s_level", "temperature_status", "humidity_status")
            ) else "warning"
            actions.append({"system": "ventilation", "state": "boosted", "reason": reason})
            insert_event(
                event_type="automation_action",
                severity=severity,
                sensor_snapshot=sensor,
                sensor_context_filtered=reason,
            )
        else:
            reason = "Ventilation nominal — all air quality and climate readings within acceptable range"
            actions.append({"system": "ventilation", "state": "nominal", "reason": reason})

    if body.door_enabled:
        chickens = sensor.get("number_of_chickens") or sensor.get("chickens_inside") or 0
        door_open = sensor.get("door_open", False)
        critical = get_critical_alerts(sensor)

        if critical:
            reason = "Door kept closed — critical conditions detected: " + "; ".join(critical)
            state = "closed"
        elif chickens == 0:
            reason = "Door can open — no chickens detected outside"
            state = "open"
        else:
            reason = f"Monitoring door — {chickens} chicken(s) still inside"
            state = "monitoring"

        actions.append({"system": "door", "state": state, "reason": reason})
        if critical:
            insert_event(
                event_type="automation_action",
                severity="warning",
                sensor_snapshot=sensor,
                sensor_context_filtered=reason,
            )

    return {
        "actions": actions,
        "evaluated_at": __import__("datetime").datetime.utcnow().isoformat(),
    }


# Serve uploaded heatmap images as static files
app.mount("/static/uploads", StaticFiles(directory="uploads"), name="uploads")


# ---------------------------------------------------------------------------
# Serve the React frontend (must be LAST — after all API routes)
# Only active once you've run `npm run build` inside frontend/
# ---------------------------------------------------------------------------

if os.path.exists("frontend/dist"):
    app.mount("/", StaticFiles(directory="frontend/dist", html=True), name="frontend")
