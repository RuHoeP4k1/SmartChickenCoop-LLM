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
    Get the most recent sensor reading, merged with the latest CV counts.

    Queries sensor_readings_colson and LEFT JOINs the latest row from
    cv_counts_colson so callers get a single unified dict.
    chickens_inside is aliased from number_of_chickens for backwards compat.

    Returns:
        Dictionary with sensor + cv data, or None if no data
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
                s.heat_stress_index,
                s.feeder_status, s.waterer_status,
                s.feeder_pct, s.waterer_pct,
                s.h2s_ppm, s.h2s_level,
                s.mold_risk_score, s.mold_risk_status,
                s.door_open, s.ventilation_on,
                s.error,
                cv.number_of_chickens,
                cv.number_of_chickens AS chickens_inside,
                cv.egg_count
            FROM sensor_readings_colson s
            LEFT JOIN LATERAL (
                SELECT number_of_chickens, egg_count
                FROM cv_counts_colson
                ORDER BY timestamp DESC
                LIMIT 1
            ) cv ON true
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
                heat_stress_index,
                feeder_status, waterer_status,
                feeder_pct, waterer_pct,
                h2s_ppm, h2s_level,
                mold_risk_score, mold_risk_status,
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
                   s.heat_stress_index,
                   s.feeder_pct, s.waterer_pct,
                   s.h2s_ppm, s.h2s_level,
                   s.mold_risk_score, s.mold_risk_status,
                   s.door_open, s.ventilation_on,
                   s.error,
                   cv.number_of_chickens
            FROM sensor_readings_colson s
            LEFT JOIN LATERAL (
                SELECT number_of_chickens
                FROM cv_counts_colson
                WHERE ABS(EXTRACT(EPOCH FROM (timestamp - s.timestamp))) <= 300
                ORDER BY ABS(EXTRACT(EPOCH FROM (timestamp - s.timestamp)))
                LIMIT 1
            ) cv ON true
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

        query = """
            INSERT INTO sensor_readings_colson (
                timestamp,
                temperature_c, temperature_status,
                humidity_pct, humidity_status,
                heat_stress_index,
                feeder_status, waterer_status,
                feeder_pct, waterer_pct,
                h2s_ppm, h2s_level,
                mold_risk_score, mold_risk_status,
                door_open, ventilation_on,
                error
            ) VALUES (
                %s, %s, %s, %s, %s, %s, %s, %s,
                %s, %s, %s, %s, %s, %s, %s, %s, %s
            )
            RETURNING id
        """

        values = (
            sensor_data.get('timestamp', datetime.now()),
            sensor_data.get('temperature_c'),
            sensor_data.get('temperature_status', 'normal'),
            sensor_data.get('humidity_pct'),
            sensor_data.get('humidity_status', 'normal'),
            sensor_data.get('heat_stress_index', 'normal'),
            sensor_data.get('feeder_status', 'full'),
            sensor_data.get('waterer_status', 'full'),
            sensor_data.get('feeder_pct'),
            sensor_data.get('waterer_pct'),
            sensor_data.get('h2s_ppm'),
            sensor_data.get('h2s_level', 'normal'),
            sensor_data.get('mold_risk_score'),
            sensor_data.get('mold_risk_status', 'normal'),
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
    mold_risk_score FLOAT,
    mold_risk_status TEXT DEFAULT 'normal',
    door_open BOOLEAN DEFAULT FALSE,
    ventilation_on BOOLEAN DEFAULT FALSE,
    error TEXT
);

CREATE INDEX IF NOT EXISTS idx_sensor_colson_timestamp ON sensor_readings_colson(timestamp DESC);
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
        cursor.execute(CREATE_CV_COUNTS_SQL)
        cursor.execute(CREATE_EVENT_LOG_SQL)
        cursor.execute(MIGRATE_EVENT_LOG_SQL)
        cursor.execute(CREATE_RESPONSE_REVIEWS_SQL)
        conn.commit()
        cursor.close()
        print("Database tables created successfully (sensor_readings_colson + cv_counts_colson + event_log + response_reviews)")
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
            print(f"   Heat stress: {latest['heat_stress_index']}")
        else:
            print("\nNo sensor readings yet. Run: python scripts/generate_demo_data.py")

    except Exception as e:
        print(f"Database error: {e}")
        print("\nMake sure PostgreSQL is running and .env has correct DB credentials!")
