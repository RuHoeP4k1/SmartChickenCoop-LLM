"""
Database utilities for sensor data and event logging
Uses psycopg2 for simple PostgreSQL queries
"""

import os
import json
import atexit
import psycopg2
import psycopg2.pool
from psycopg2.extras import RealDictCursor
from typing import Dict, Optional, List
from datetime import datetime, timedelta
from dotenv import load_dotenv

load_dotenv()

# ── Connection config ──────────────────────────────────────────────────────────
# Priority: DATABASE_URL (Supabase / any hosted Postgres)
#           → individual DB_* vars (local Postgres)
#
# To switch to Supabase: set DATABASE_URL in .env and leave DB_* as fallbacks.
# To switch back to local: clear DATABASE_URL (or comment it out).
_DATABASE_URL = os.getenv("DATABASE_URL", "").strip()

print("[DB] Initializing connection pool...")
try:
    if _DATABASE_URL:
        # Hosted Postgres / Supabase — single connection string
        # connect_timeout prevents blocking uvicorn startup if DB is unreachable
        _dsn = _DATABASE_URL if "connect_timeout" in _DATABASE_URL else _DATABASE_URL + "?connect_timeout=5"
        _pool = psycopg2.pool.SimpleConnectionPool(minconn=1, maxconn=5, dsn=_dsn)
        print(f"[DB] Connected via DATABASE_URL (Supabase / hosted Postgres)")
    else:
        # Local Postgres — individual vars
        DB_CONFIG = {
            "host": os.getenv("DB_HOST", "localhost"),
            "database": os.getenv("DB_NAME", "chickens"),
            "user": os.getenv("DB_USER", "postgres"),
            "password": os.getenv("DB_PASSWORD", ""),
            "port": int(os.getenv("DB_PORT", "5432")),
            "connect_timeout": 5,
        }
        _pool = psycopg2.pool.SimpleConnectionPool(minconn=1, maxconn=5, **DB_CONFIG)
        print(f"[DB] Connected to local Postgres ({DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['database']})")
except Exception as _db_init_err:
    print(f"[DB] WARNING: Pool creation failed at import time: {_db_init_err}. DB calls will fail until reconnected.")
    _pool = None
print(f"[DB] Pool init complete. pool={'ok' if _pool else 'FAILED'}")

atexit.register(lambda: _pool.closeall() if _pool and not _pool.closed else None)


def get_db_connection():
    """
    Get a connection from the pool.
    IMPORTANT: callers must call release_db_connection(conn) when done
    instead of conn.close().
    """
    if _pool is None:
        raise RuntimeError("Database pool not initialized — check DB credentials/connectivity.")
    return _pool.getconn()


def release_db_connection(conn):
    """Return a connection to the pool."""
    _pool.putconn(conn)


def get_latest_sensor_reading() -> Optional[Dict]:
    """
    Get the most recent sensor reading, merged with the latest CV counts
    AND the latest risk snapshot (heat + mold).

    The Pi pipeline writes risk_snapshots from a separate process; the
    legacy heat_stress_index / mold_risk_* columns on sensor_readings_colson
    are kept nullable for backwards compat but no longer authoritative.

    Returns:
        Dictionary with sensor + cv + risk data, or None if no data
    """
    conn = get_db_connection()
    try:
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        cursor.execute(
            """
            SELECT
                s.id, s.timestamp,
                s.temperature_c, s.temperature_status,
                s.humidity_pct, s.humidity_status,
                s.feeder_status, s.waterer_status,
                s.feeder_pct, s.waterer_pct,
                s.h2s_ppm, s.h2s_level,
                s.co2_ppm, s.co2_level,
                s.nh3_ppm, s.nh3_level,
                s.door_open, s.ventilation_on,
                s.error,
                cv.number_of_chickens,
                cv.number_of_chickens AS chickens_inside,
                cv.egg_count,
                r.heat_risk_score,
                r.heat_risk_level,
                r.thi_current,
                r.mold_risk_score,
                r.mold_risk_level,
                r.contributing_factors AS risk_factors,
                c.crowding_verdict
            FROM sensor_readings_colson s
            LEFT JOIN LATERAL (
                SELECT number_of_chickens, egg_count
                FROM cv_counts_colson
                ORDER BY timestamp DESC
                LIMIT 1
            ) cv ON true
            LEFT JOIN LATERAL (
                SELECT heat_risk_score, heat_risk_level, thi_current,
                       mold_risk_score, mold_risk_level, contributing_factors
                FROM risk_snapshots
                ORDER BY created_at DESC
                LIMIT 1
            ) r ON true
            LEFT JOIN LATERAL (
                SELECT crowding_verdict
                FROM crowding
                ORDER BY created_at DESC
                LIMIT 1
            ) c ON true
            ORDER BY s.timestamp DESC
            LIMIT 1
            """
        )
        result = cursor.fetchone()
        cursor.close()
        return dict(result) if result else None
    finally:
        release_db_connection(conn)


def get_recent_readings(limit: int = 50) -> List[Dict]:
    """
    Get the last N sensor readings, ordered newest first.

    Args:
        limit: Number of rows to return

    Returns:
        List of sensor reading dictionaries
    """
    conn = get_db_connection()
    try:
        cursor = conn.cursor(cursor_factory=RealDictCursor)

        query = """
            SELECT
                id, timestamp,
                temperature_c, temperature_status,
                humidity_pct, humidity_status,
                feeder_status, waterer_status,
                feeder_pct, waterer_pct,
                h2s_ppm, h2s_level,
                co2_ppm, co2_level,
                nh3_ppm, nh3_level,
                door_open, ventilation_on,
                error
            FROM sensor_readings_colson
            ORDER BY id DESC
            LIMIT %s
        """

        cursor.execute(query, (limit,))
        results = cursor.fetchall()
        cursor.close()

        return [dict(row) for row in results]
    finally:
        release_db_connection(conn)


def get_sensor_history(hours: float = 1, limit: int = 200) -> List[Dict]:
    """
    Get sensor readings from the past N hours, oldest first.
    Used by the frontend chart to show historical trends.

    Args:
        hours: How many hours back to query
        limit: Max rows to return (server-side downsampling)

    Returns:
        List of sensor reading dicts ordered by timestamp ASC
    """
    conn = get_db_connection()
    try:
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        since = datetime.now() - timedelta(hours=hours)
        cursor.execute(
            """
            SELECT s.timestamp,
                   s.temperature_c, s.temperature_status,
                   s.humidity_pct, s.humidity_status,
                   s.feeder_pct, s.waterer_pct,
                   s.h2s_ppm, s.h2s_level,
                   s.door_open, s.ventilation_on,
                   s.error,
                   cv.number_of_chickens,
                   r.heat_risk_score, r.heat_risk_level,
                   r.thi_current,
                   r.mold_risk_score, r.mold_risk_level
            FROM sensor_readings_colson s
            LEFT JOIN LATERAL (
                SELECT number_of_chickens
                FROM cv_counts_colson
                WHERE ABS(EXTRACT(EPOCH FROM (timestamp - s.timestamp))) <= 300
                ORDER BY ABS(EXTRACT(EPOCH FROM (timestamp - s.timestamp)))
                LIMIT 1
            ) cv ON true
            LEFT JOIN LATERAL (
                SELECT heat_risk_score, heat_risk_level, thi_current,
                       mold_risk_score, mold_risk_level
                FROM risk_snapshots
                WHERE ABS(EXTRACT(EPOCH FROM (created_at - s.timestamp))) <= 600
                ORDER BY ABS(EXTRACT(EPOCH FROM (created_at - s.timestamp)))
                LIMIT 1
            ) r ON true
            WHERE s.timestamp >= %s
            ORDER BY s.timestamp ASC
            LIMIT %s
            """,
            (since, limit),
        )
        rows = cursor.fetchall()
        cursor.close()
        return [dict(r) for r in rows]
    finally:
        release_db_connection(conn)


def insert_sensor_reading(sensor_data: Dict) -> int:
    """
    Insert a new sensor reading into the database.

    Used by Raspberry Pi to store readings.

    Args:
        sensor_data: Dictionary with sensor values

    Returns:
        ID of inserted row
    """
    conn = get_db_connection()
    try:
        cursor = conn.cursor()

        # NOTE: heat_stress_index and mold_risk_score/status are kept nullable
        # for backwards compat with the Pi pipeline. Authoritative risk data
        # now lives in risk_snapshots (see insert_risk_snapshot).
        query = """
            INSERT INTO sensor_readings_colson (
                timestamp,
                temperature_c, temperature_status,
                humidity_pct, humidity_status,
                feeder_status, waterer_status,
                feeder_pct, waterer_pct,
                h2s_ppm, h2s_level,
                door_open, ventilation_on,
                error
            ) VALUES (
                %s, %s, %s, %s, %s, %s, %s,
                %s, %s, %s, %s, %s, %s, %s
            )
            RETURNING id
        """

        values = (
            sensor_data.get('timestamp', datetime.now()),
            sensor_data.get('temperature_c'),
            sensor_data.get('temperature_status', 'normal'),
            sensor_data.get('humidity_pct'),
            sensor_data.get('humidity_status', 'normal'),
            sensor_data.get('feeder_status', 'full'),
            sensor_data.get('waterer_status', 'full'),
            sensor_data.get('feeder_pct'),
            sensor_data.get('waterer_pct'),
            sensor_data.get('h2s_ppm'),
            sensor_data.get('h2s_level', 'normal'),
            sensor_data.get('door_open', False),
            sensor_data.get('ventilation_on', False),
            sensor_data.get('error'),
        )

        cursor.execute(query, values)
        reading_id = cursor.fetchone()[0]

        conn.commit()
        cursor.close()

        return reading_id
    except Exception:
        conn.rollback()
        raise
    finally:
        release_db_connection(conn)


# =============================================================================
# CV COUNTS
# =============================================================================

def insert_cv_count(chickens: int, eggs: int) -> int:
    """Insert a new row into cv_counts_colson. Returns the new row id."""
    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO cv_counts_colson (timestamp, number_of_chickens, egg_count)
            VALUES (%s, %s, %s)
            RETURNING id
            """,
            (datetime.now(), chickens, eggs),
        )
        row_id = cursor.fetchone()[0]
        conn.commit()
        cursor.close()
        return row_id
    except Exception:
        conn.rollback()
        raise
    finally:
        release_db_connection(conn)


def get_latest_cv_count() -> Optional[tuple]:
    """
    Return (number_of_chickens, egg_count) of the most recent row,
    or None if the table is empty.
    """
    conn = get_db_connection()
    try:
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        cursor.execute(
            """
            SELECT number_of_chickens, egg_count
            FROM cv_counts_colson
            ORDER BY timestamp DESC
            LIMIT 1
            """
        )
        row = cursor.fetchone()
        cursor.close()
        if row is None:
            return None
        return (row["number_of_chickens"], row["egg_count"])
    finally:
        release_db_connection(conn)


# =============================================================================
# EVENT LOG TABLE — tracks LLM responses + sensor alerts for evaluation
# =============================================================================

CREATE_EVENT_LOG_SQL = """
CREATE TABLE IF NOT EXISTS event_log (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    event_type TEXT NOT NULL,
    severity TEXT DEFAULT 'info',
    user_query TEXT,
    llm_response TEXT,
    sensor_snapshot JSONB,
    sensor_context_filtered TEXT,
    sources JSONB
);

CREATE INDEX IF NOT EXISTS idx_event_timestamp ON event_log(timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_event_type ON event_log(event_type);
"""

MIGRATE_EVENT_LOG_SQL = """
ALTER TABLE event_log ADD COLUMN IF NOT EXISTS sources JSONB;
ALTER TABLE event_log ADD COLUMN IF NOT EXISTS routing_mode TEXT;
ALTER TABLE event_log ADD COLUMN IF NOT EXISTS routing_decision TEXT;
ALTER TABLE event_log ADD COLUMN IF NOT EXISTS prompt_template TEXT;
ALTER TABLE event_log ADD COLUMN IF NOT EXISTS response_time_ms INTEGER;
"""

MIGRATE_SENSOR_READINGS_SQL = """
ALTER TABLE sensor_readings_colson ADD COLUMN IF NOT EXISTS co2_ppm DOUBLE PRECISION;
ALTER TABLE sensor_readings_colson ADD COLUMN IF NOT EXISTS co2_level TEXT DEFAULT 'normal';
ALTER TABLE sensor_readings_colson ADD COLUMN IF NOT EXISTS nh3_ppm DOUBLE PRECISION;
ALTER TABLE sensor_readings_colson ADD COLUMN IF NOT EXISTS nh3_level TEXT DEFAULT 'normal';
"""


def insert_event(
    event_type: str,
    severity: str = "info",
    user_query: str = None,
    llm_response: str = None,
    sensor_snapshot: dict = None,
    sensor_context_filtered: str = None,
    sources: list = None,
    routing_mode: str = None,
    routing_decision: str = None,
    prompt_template: str = None,
    response_time_ms: int = None,
) -> int:
    """
    Log an event to the event_log table.

    Used for:
    - LLM responses (event_type='llm_response'): logs query + answer + sensor state
    - Sensor alerts (event_type='sensor_alert'): logs threshold breaches
    - Normal status (event_type='conditions_normal'): logs when conditions clear

    Returns:
        ID of inserted event row
    """
    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        cursor.execute(
            """INSERT INTO event_log
               (event_type, severity, user_query, llm_response, sensor_snapshot,
                sensor_context_filtered, sources, routing_mode, routing_decision,
                prompt_template, response_time_ms)
               VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s) RETURNING id""",
            (
                event_type,
                severity,
                user_query,
                llm_response,
                json.dumps(sensor_snapshot, default=str) if sensor_snapshot else None,
                sensor_context_filtered,
                json.dumps(list(dict.fromkeys(sources))) if sources else None,
                routing_mode,
                routing_decision,
                prompt_template,
                response_time_ms,
            ),
        )
        event_id = cursor.fetchone()[0]
        conn.commit()
        cursor.close()
        return event_id
    except Exception:
        conn.rollback()
        raise
    finally:
        release_db_connection(conn)


def get_recent_events(limit: int = 20, event_type: str = None) -> List[Dict]:
    """
    Get recent events from the log, newest first.

    Args:
        limit: Max rows to return
        event_type: Optional filter (e.g. 'llm_response', 'sensor_alert')

    Returns:
        List of event dicts
    """
    conn = get_db_connection()
    try:
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        if event_type:
            cursor.execute(
                "SELECT * FROM event_log WHERE event_type = %s ORDER BY id DESC LIMIT %s",
                (event_type, limit),
            )
        else:
            cursor.execute(
                "SELECT * FROM event_log ORDER BY id DESC LIMIT %s",
                (limit,),
            )
        rows = cursor.fetchall()
        cursor.close()
        return [dict(r) for r in rows]
    finally:
        release_db_connection(conn)


# =============================================================================
# DATABASE SETUP
# =============================================================================

CREATE_RISK_SNAPSHOTS_SQL = """
CREATE TABLE IF NOT EXISTS risk_snapshots (
    id BIGSERIAL PRIMARY KEY,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    heat_risk_score SMALLINT,
    heat_risk_level TEXT,
    thi_current DOUBLE PRECISION,
    high_thi_streak_minutes INTEGER,
    mold_risk_score SMALLINT,
    mold_risk_level TEXT,
    mold_favourable_for_growth BOOLEAN,
    mold_index_m INTEGER,
    mold_consecutive_unfavourable_minutes INTEGER,
    contributing_factors TEXT
);

CREATE INDEX IF NOT EXISTS idx_risk_snapshots_created_at ON risk_snapshots(created_at DESC);
"""

CREATE_EGG_CALENDAR_SQL = """
CREATE TABLE IF NOT EXISTS egg_calendar_entries (
    entry_date DATE PRIMARY KEY,
    eggs_laid INTEGER NOT NULL DEFAULT 0,
    source TEXT NOT NULL DEFAULT 'auto',  -- 'auto' | 'manual'
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
"""

CREATE_CHORE_DEFINITIONS_SQL = """
CREATE TABLE IF NOT EXISTS chore_definitions (
    id SERIAL PRIMARY KEY,
    label TEXT NOT NULL UNIQUE,
    checklist_items JSONB DEFAULT '[]'::jsonb,
    llm_template TEXT,
    is_active BOOLEAN DEFAULT TRUE
);
"""

CREATE_CHORE_LOG_SQL = """
CREATE TABLE IF NOT EXISTS chore_log (
    id SERIAL PRIMARY KEY,
    entry_date DATE NOT NULL,
    definition_id INTEGER REFERENCES chore_definitions(id) ON DELETE CASCADE,
    checked_items JSONB DEFAULT '[]'::jsonb,
    notes TEXT,
    completed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_chore_log_date ON chore_log(entry_date DESC);
"""

CREATE_AUTOMATION_WINDOWS_SQL = """
CREATE TABLE IF NOT EXISTS automation_windows (
    id SERIAL PRIMARY KEY,
    task TEXT NOT NULL,  -- 'ventilation' | 'door' | 'feeder' (app-enforced)
    start_date DATE NOT NULL,
    end_date DATE NOT NULL,
    start_time TIME,  -- NULL = all day
    end_time TIME,    -- NULL = all day
    days_of_week INTEGER[] DEFAULT '{0,1,2,3,4,5,6}'::int[],  -- 0=Sun..6=Sat
    enabled BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_automation_windows_range ON automation_windows(start_date, end_date);
"""

_DEFAULT_CHORES = [
    ("Cleaned coop",
     ["bedding", "waterer", "feeder", "nestbox"],
     "Coop was cleaned {days_ago} days ago ({items})."),
    ("Deep clean",
     ["scrubbed walls", "replaced all bedding", "disinfected"],
     "Coop had a deep clean {days_ago} days ago ({items})."),
    ("Refilled feed",
     [],
     "Feed was refilled {days_ago} days ago."),
    ("Refilled water",
     [],
     "Water was refilled {days_ago} days ago."),
    ("Health check",
     ["eyes", "comb", "vent", "feet"],
     "Flock health check done {days_ago} days ago ({items})."),
    ("Treated for parasites",
     [],
     "Parasite treatment applied {days_ago} days ago."),
]


def _seed_default_chores(cursor) -> None:
    """Insert the default chore definitions if the table is empty."""
    cursor.execute("SELECT COUNT(*) FROM chore_definitions")
    if cursor.fetchone()[0] > 0:
        return
    for label, items, tpl in _DEFAULT_CHORES:
        cursor.execute(
            """INSERT INTO chore_definitions (label, checklist_items, llm_template)
               VALUES (%s, %s::jsonb, %s)
               ON CONFLICT (label) DO NOTHING""",
            (label, json.dumps(items), tpl),
        )


# =============================================================================
# RISK SNAPSHOTS
# =============================================================================

def insert_risk_snapshot(
    heat_risk_score: Optional[float] = None,
    heat_risk_level: Optional[str] = None,
    thi_current: Optional[float] = None,
    high_thi_streak_minutes: Optional[int] = None,
    mold_risk_score: Optional[float] = None,
    mold_risk_level: Optional[str] = None,
    mold_favourable_for_growth: Optional[bool] = None,
    mold_index_m: Optional[int] = None,
    mold_consecutive_unfavourable_minutes: Optional[int] = None,
    contributing_factors: Optional[str] = None,
) -> int:
    """Insert a risk snapshot row. Used by simulation and by the Pi pipeline."""
    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        cursor.execute(
            """INSERT INTO risk_snapshots (
                   heat_risk_score, heat_risk_level, thi_current,
                   high_thi_streak_minutes,
                   mold_risk_score, mold_risk_level,
                   mold_favourable_for_growth, mold_index_m,
                   mold_consecutive_unfavourable_minutes,
                   contributing_factors
               ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
               RETURNING id""",
            (
                int(round(float(heat_risk_score))) if heat_risk_score is not None else None,
                heat_risk_level,
                thi_current,
                high_thi_streak_minutes,
                int(round(float(mold_risk_score))) if mold_risk_score is not None else None,
                mold_risk_level,
                mold_favourable_for_growth,
                mold_index_m,
                mold_consecutive_unfavourable_minutes,
                contributing_factors,
            ),
        )
        row_id = cursor.fetchone()[0]
        conn.commit()
        cursor.close()
        return row_id
    except Exception:
        conn.rollback()
        raise
    finally:
        release_db_connection(conn)


def get_latest_risk_snapshot() -> Optional[Dict]:
    """Return the most recent row from risk_snapshots including all columns."""
    conn = get_db_connection()
    try:
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        cursor.execute(
            "SELECT * FROM risk_snapshots ORDER BY created_at DESC LIMIT 1"
        )
        row = cursor.fetchone()
        cursor.close()
        return dict(row) if row else None
    finally:
        release_db_connection(conn)


# =============================================================================
# EGG CALENDAR
# =============================================================================

def upsert_egg_entry(entry_date, eggs_laid: int, source: str = "auto") -> None:
    """
    Insert or update an egg calendar entry. Manual edits win over automatic
    reconciliation: if an entry exists with source='manual' and the incoming
    source is 'auto', the existing row is left untouched.
    """
    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        if source == "auto":
            cursor.execute(
                """INSERT INTO egg_calendar_entries (entry_date, eggs_laid, source, updated_at)
                   VALUES (%s, %s, 'auto', CURRENT_TIMESTAMP)
                   ON CONFLICT (entry_date) DO UPDATE SET
                       eggs_laid = CASE
                           WHEN egg_calendar_entries.source = 'manual' THEN egg_calendar_entries.eggs_laid
                           ELSE EXCLUDED.eggs_laid
                       END,
                       updated_at = CURRENT_TIMESTAMP""",
                (entry_date, eggs_laid),
            )
        else:
            cursor.execute(
                """INSERT INTO egg_calendar_entries (entry_date, eggs_laid, source, updated_at)
                   VALUES (%s, %s, 'manual', CURRENT_TIMESTAMP)
                   ON CONFLICT (entry_date) DO UPDATE SET
                       eggs_laid = EXCLUDED.eggs_laid,
                       source = 'manual',
                       updated_at = CURRENT_TIMESTAMP""",
                (entry_date, eggs_laid),
            )
        conn.commit()
        cursor.close()
    except Exception:
        conn.rollback()
        raise
    finally:
        release_db_connection(conn)


def get_egg_entries_for_month(year: int, month: int) -> List[Dict]:
    """Return all egg calendar entries in the given month."""
    conn = get_db_connection()
    try:
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        cursor.execute(
            """SELECT entry_date, eggs_laid, source, updated_at
               FROM egg_calendar_entries
               WHERE EXTRACT(YEAR FROM entry_date) = %s
                 AND EXTRACT(MONTH FROM entry_date) = %s
               ORDER BY entry_date ASC""",
            (year, month),
        )
        rows = cursor.fetchall()
        cursor.close()
        return [dict(r) for r in rows]
    finally:
        release_db_connection(conn)


def get_cv_egg_counts_for_date(target_date) -> List[Dict]:
    """Return all cv_counts_colson egg_count samples for a given date, ordered by time."""
    conn = get_db_connection()
    try:
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        cursor.execute(
            """SELECT timestamp, egg_count
               FROM cv_counts_colson
               WHERE timestamp::date = %s
               ORDER BY timestamp ASC""",
            (target_date,),
        )
        rows = cursor.fetchall()
        cursor.close()
        return [dict(r) for r in rows]
    finally:
        release_db_connection(conn)


# =============================================================================
# CHORES
# =============================================================================

def get_chore_definitions() -> List[Dict]:
    """Return all active chore definitions."""
    conn = get_db_connection()
    try:
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        cursor.execute(
            """SELECT id, label, checklist_items, llm_template
               FROM chore_definitions
               WHERE is_active = TRUE
               ORDER BY id ASC"""
        )
        rows = cursor.fetchall()
        cursor.close()
        return [dict(r) for r in rows]
    finally:
        release_db_connection(conn)


def insert_chore_log(
    entry_date,
    definition_id: int,
    checked_items: Optional[List[str]] = None,
    notes: str = "",
) -> int:
    """Record a completed chore."""
    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        cursor.execute(
            """INSERT INTO chore_log (entry_date, definition_id, checked_items, notes)
               VALUES (%s, %s, %s::jsonb, %s)
               RETURNING id""",
            (entry_date, definition_id, json.dumps(checked_items or []), notes),
        )
        log_id = cursor.fetchone()[0]
        conn.commit()
        cursor.close()
        return log_id
    except Exception:
        conn.rollback()
        raise
    finally:
        release_db_connection(conn)


def delete_chore_log(log_id: int) -> int:
    """Remove a chore log entry. Returns rows deleted (0 or 1)."""
    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM chore_log WHERE id = %s", (log_id,))
        rowcount = cursor.rowcount
        conn.commit()
        cursor.close()
        return rowcount
    except Exception:
        conn.rollback()
        raise
    finally:
        release_db_connection(conn)


def get_chore_log_in_range(start_date, end_date) -> List[Dict]:
    """Return chore log entries between start_date and end_date inclusive, joined with definitions."""
    conn = get_db_connection()
    try:
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        cursor.execute(
            """SELECT cl.id, cl.entry_date, cl.definition_id, cl.checked_items,
                      cl.notes, cl.completed_at,
                      cd.label, cd.llm_template
               FROM chore_log cl
               JOIN chore_definitions cd ON cd.id = cl.definition_id
               WHERE cl.entry_date BETWEEN %s AND %s
               ORDER BY cl.entry_date DESC, cl.completed_at DESC""",
            (start_date, end_date),
        )
        rows = cursor.fetchall()
        cursor.close()
        return [dict(r) for r in rows]
    finally:
        release_db_connection(conn)


# =============================================================================
# AUTOMATION WINDOWS
# =============================================================================

ALLOWED_AUTOMATION_TASKS = ("ventilation", "door", "feeder")


def insert_automation_window(
    task: str,
    start_date,
    end_date,
    start_time=None,
    end_time=None,
    days_of_week: Optional[List[int]] = None,
) -> int:
    """Create an automation window. Raises ValueError on invalid task."""
    if task not in ALLOWED_AUTOMATION_TASKS:
        raise ValueError(f"task must be one of {ALLOWED_AUTOMATION_TASKS}")
    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        cursor.execute(
            """INSERT INTO automation_windows
                   (task, start_date, end_date, start_time, end_time, days_of_week)
               VALUES (%s, %s, %s, %s, %s, %s)
               RETURNING id""",
            (
                task, start_date, end_date, start_time, end_time,
                days_of_week if days_of_week is not None else [0, 1, 2, 3, 4, 5, 6],
            ),
        )
        win_id = cursor.fetchone()[0]
        conn.commit()
        cursor.close()
        return win_id
    except Exception:
        conn.rollback()
        raise
    finally:
        release_db_connection(conn)


def delete_automation_window(win_id: int) -> int:
    """Remove an automation window. Returns rows deleted (0 or 1)."""
    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM automation_windows WHERE id = %s", (win_id,))
        rowcount = cursor.rowcount
        conn.commit()
        cursor.close()
        return rowcount
    except Exception:
        conn.rollback()
        raise
    finally:
        release_db_connection(conn)


def get_automation_windows_in_range(start_date, end_date) -> List[Dict]:
    """Return automation windows that overlap the given date range."""
    conn = get_db_connection()
    try:
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        cursor.execute(
            """SELECT id, task, start_date, end_date, start_time, end_time,
                      days_of_week, enabled, created_at
               FROM automation_windows
               WHERE enabled = TRUE
                 AND start_date <= %s
                 AND end_date >= %s
               ORDER BY start_date ASC""",
            (end_date, start_date),
        )
        rows = cursor.fetchall()
        cursor.close()
        return [dict(r) for r in rows]
    finally:
        release_db_connection(conn)


def get_feeder_waterer_samples_for_date(target_date) -> List[Dict]:
    """Return feeder_pct/waterer_pct samples for a given date, ordered by time."""
    conn = get_db_connection()
    try:
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        cursor.execute(
            """SELECT timestamp, feeder_pct, waterer_pct
               FROM sensor_readings_colson
               WHERE timestamp::date = %s
                 AND (feeder_pct IS NOT NULL OR waterer_pct IS NOT NULL)
               ORDER BY timestamp ASC""",
            (target_date,),
        )
        rows = cursor.fetchall()
        cursor.close()
        return [dict(r) for r in rows]
    finally:
        release_db_connection(conn)


# =============================================================================
# LEGACY sensor_readings_colson + cv_counts_colson CREATE
# =============================================================================

CREATE_SENSOR_READINGS_COLSON_SQL = """
CREATE TABLE IF NOT EXISTS sensor_readings_colson (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    temperature_c FLOAT,
    temperature_status TEXT DEFAULT 'normal',
    humidity_pct FLOAT,
    humidity_status TEXT DEFAULT 'normal',
    heat_stress_index TEXT DEFAULT 'normal',
    feeder_status TEXT DEFAULT 'full',
    waterer_status TEXT DEFAULT 'full',
    feeder_pct FLOAT,
    waterer_pct FLOAT,
    h2s_ppm FLOAT,
    h2s_level TEXT DEFAULT 'normal',
    co2_ppm FLOAT,
    co2_level TEXT DEFAULT 'normal',
    nh3_ppm FLOAT,
    nh3_level TEXT DEFAULT 'normal',
    mold_risk_score FLOAT,
    mold_risk_status TEXT DEFAULT 'normal',
    door_open BOOLEAN DEFAULT FALSE,
    ventilation_on BOOLEAN DEFAULT FALSE,
    error TEXT
);

CREATE INDEX IF NOT EXISTS idx_sensor_colson_timestamp ON sensor_readings_colson(timestamp DESC);
"""

CREATE_CROWDING_SQL = """
CREATE TABLE IF NOT EXISTS crowding (
    id SERIAL PRIMARY KEY,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    crowding_verdict TEXT
);

CREATE INDEX IF NOT EXISTS idx_crowding_created_at ON crowding(created_at DESC);
"""

CREATE_CV_COUNTS_SQL = """
CREATE TABLE IF NOT EXISTS cv_counts_colson (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    number_of_chickens INT,
    egg_count INT
);

CREATE INDEX IF NOT EXISTS idx_cv_colson_timestamp ON cv_counts_colson(timestamp DESC);
"""


CREATE_RESPONSE_REVIEWS_SQL = """
CREATE TABLE IF NOT EXISTS response_reviews (
    id SERIAL PRIMARY KEY,
    event_id INTEGER REFERENCES event_log(id) ON DELETE CASCADE,
    is_good BOOLEAN NOT NULL,
    routing_correct BOOLEAN,
    notes TEXT,
    reviewed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(event_id)
);
"""


def upsert_review(
    event_id: int,
    is_good: bool,
    routing_correct: bool = None,
    notes: str = "",
) -> int:
    """Insert or update a review for an event_log entry."""
    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        cursor.execute(
            """INSERT INTO response_reviews (event_id, is_good, routing_correct, notes)
               VALUES (%s, %s, %s, %s)
               ON CONFLICT (event_id) DO UPDATE SET
                   is_good = EXCLUDED.is_good,
                   routing_correct = EXCLUDED.routing_correct,
                   notes = EXCLUDED.notes,
                   reviewed_at = CURRENT_TIMESTAMP
               RETURNING id""",
            (event_id, is_good, routing_correct, notes),
        )
        review_id = cursor.fetchone()[0]
        conn.commit()
        cursor.close()
        return review_id
    except Exception:
        conn.rollback()
        raise
    finally:
        release_db_connection(conn)


def get_events_for_review(
    limit: int = 50,
    reviewed: bool = None,
) -> List[Dict]:
    """
    Get llm_response events with review status.

    Args:
        limit: Max rows
        reviewed: None=all, True=only reviewed, False=only unreviewed
    """
    conn = get_db_connection()
    try:
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        base = """
            SELECT e.*, r.id AS review_id, r.is_good, r.routing_correct,
                   r.notes AS review_notes, r.reviewed_at
            FROM event_log e
            LEFT JOIN response_reviews r ON r.event_id = e.id
            WHERE e.event_type = 'llm_response'
        """
        if reviewed is True:
            base += " AND r.id IS NOT NULL"
        elif reviewed is False:
            base += " AND r.id IS NULL"
        base += " ORDER BY e.id DESC LIMIT %s"
        cursor.execute(base, (limit,))
        rows = cursor.fetchall()
        cursor.close()
        return [dict(r) for r in rows]
    finally:
        release_db_connection(conn)


def export_reviews() -> List[Dict]:
    """Export all llm_response events with reviews for paper analysis."""
    conn = get_db_connection()
    try:
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        cursor.execute("""
            SELECT e.id, e.timestamp, e.severity, e.user_query, e.llm_response,
                   e.sensor_snapshot, e.sensor_context_filtered, e.sources,
                   e.routing_mode, e.routing_decision, e.prompt_template,
                   e.response_time_ms,
                   r.is_good, r.routing_correct, r.notes AS review_notes,
                   r.reviewed_at
            FROM event_log e
            LEFT JOIN response_reviews r ON r.event_id = e.id
            WHERE e.event_type = 'llm_response'
            ORDER BY e.id
        """)
        rows = cursor.fetchall()
        cursor.close()
        return [dict(r) for r in rows]
    finally:
        release_db_connection(conn)


def setup_database():
    """
    Create all tables if they don't exist.
    Run this once during setup.
    """
    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        cursor.execute(CREATE_SENSOR_READINGS_COLSON_SQL)
        cursor.execute(MIGRATE_SENSOR_READINGS_SQL)
        cursor.execute(CREATE_CROWDING_SQL)
        cursor.execute(CREATE_CV_COUNTS_SQL)
        cursor.execute(CREATE_RISK_SNAPSHOTS_SQL)
        cursor.execute(CREATE_EVENT_LOG_SQL)
        cursor.execute(MIGRATE_EVENT_LOG_SQL)
        cursor.execute(CREATE_RESPONSE_REVIEWS_SQL)
        cursor.execute(CREATE_EGG_CALENDAR_SQL)
        cursor.execute(CREATE_CHORE_DEFINITIONS_SQL)
        cursor.execute(CREATE_CHORE_LOG_SQL)
        cursor.execute(CREATE_AUTOMATION_WINDOWS_SQL)
        _seed_default_chores(cursor)
        conn.commit()
        cursor.close()
        print("Database tables created successfully (sensor + cv + risk_snapshots + event_log + reviews + egg/chore/automation)")
    finally:
        release_db_connection(conn)


if __name__ == "__main__":
    """
    Run this script directly to set up database tables.

    Usage:
        python db_utils.py
    For demo data, use: python scripts/generate_demo_data.py
    """
    print("Setting up database...")

    try:
        setup_database()

        latest = get_latest_sensor_reading()
        if latest:
            print(f"\nLatest sensor reading:")
            print(f"   Temperature: {latest['temperature_c']}C [{latest['temperature_status']}]")
            print(f"   Humidity: {latest['humidity_pct']}% [{latest['humidity_status']}]")
            print(f"   Heat risk: {latest.get('heat_risk_level', 'n/a')}")
        else:
            print("\nNo sensor readings yet. Run: python scripts/generate_demo_data.py")

    except Exception as e:
        print(f"Database error: {e}")
        print("\nMake sure PostgreSQL is running and .env has correct DB credentials!")
