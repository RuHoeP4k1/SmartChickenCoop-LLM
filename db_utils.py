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

# Database connection configuration — reads from .env, falls back to defaults
DB_CONFIG = {
    "host": os.getenv("DB_HOST", "localhost"),
    "database": os.getenv("DB_NAME", "chickens"),
    "user": os.getenv("DB_USER", "postgres"),
    "password": os.getenv("DB_PASSWORD", ""),
    "port": int(os.getenv("DB_PORT", "5432"))
}

# Connection pool — reuses connections instead of opening/closing each call.
# minconn=1 keeps one connection warm; maxconn=5 is plenty for API + scheduler.
_pool = psycopg2.pool.SimpleConnectionPool(minconn=1, maxconn=5, **DB_CONFIG)
atexit.register(lambda: _pool.closeall() if _pool and not _pool.closed else None)


def get_db_connection():
    """
    Get a connection from the pool.
    IMPORTANT: callers must call release_db_connection(conn) when done
    instead of conn.close().
    """
    return _pool.getconn()


def release_db_connection(conn):
    """Return a connection to the pool."""
    _pool.putconn(conn)


def get_latest_sensor_reading() -> Optional[Dict]:
    """
    Get the most recent sensor reading from database.

    Returns:
        Dictionary with sensor data, or None if no data
    """
    conn = get_db_connection()
    try:
        cursor = conn.cursor(cursor_factory=RealDictCursor)

        query = """
            SELECT
                id,
                timestamp,
                temperature_c,
                temperature_status,
                humidity_pct,
                humidity_status,
                heat_stress_index,
                feeder_status,
                waterer_status
            FROM sensor_readings
            ORDER BY timestamp DESC
            LIMIT 1
        """

        cursor.execute(query)
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
                id,
                timestamp,
                temperature_c,
                temperature_status,
                humidity_pct,
                humidity_status,
                heat_stress_index,
                feeder_status,
                waterer_status
            FROM sensor_readings
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
            SELECT timestamp, temperature_c, temperature_status,
                   humidity_pct, humidity_status, heat_stress_index
            FROM sensor_readings
            WHERE timestamp >= %s
            ORDER BY timestamp ASC
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
            INSERT INTO sensor_readings (
                timestamp,
                temperature_c,
                temperature_status,
                humidity_pct,
                humidity_status,
                heat_stress_index,
                feeder_status,
                waterer_status
            ) VALUES (
                %s, %s, %s, %s, %s, %s, %s, %s
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
            sensor_data.get('waterer_status', 'full')
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
    sensor_context_filtered TEXT
);

CREATE INDEX IF NOT EXISTS idx_event_timestamp ON event_log(timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_event_type ON event_log(event_type);
"""


def insert_event(
    event_type: str,
    severity: str = "info",
    user_query: str = None,
    llm_response: str = None,
    sensor_snapshot: dict = None,
    sensor_context_filtered: str = None,
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
               (event_type, severity, user_query, llm_response, sensor_snapshot, sensor_context_filtered)
               VALUES (%s, %s, %s, %s, %s, %s) RETURNING id""",
            (
                event_type,
                severity,
                user_query,
                llm_response,
                json.dumps(sensor_snapshot) if sensor_snapshot else None,
                sensor_context_filtered,
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

CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS sensor_readings (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    temperature_c FLOAT,
    temperature_status TEXT DEFAULT 'normal',
    humidity_pct FLOAT,
    humidity_status TEXT DEFAULT 'normal',
    heat_stress_index TEXT DEFAULT 'normal',
    feeder_status TEXT DEFAULT 'full',
    waterer_status TEXT DEFAULT 'full'
);

CREATE INDEX IF NOT EXISTS idx_sensor_timestamp ON sensor_readings(timestamp DESC);
"""


def setup_database():
    """
    Create all tables if they don't exist.
    Run this once during setup.
    """
    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        cursor.execute(CREATE_TABLE_SQL)
        cursor.execute(CREATE_EVENT_LOG_SQL)
        conn.commit()
        cursor.close()
        print("Database tables created successfully (sensor_readings + event_log)")
    finally:
        release_db_connection(conn)


def insert_mock_data():
    """
    Insert some mock sensor data for testing.
    Run this to populate database with test data.
    """
    mock_readings = [
        {
            "timestamp": datetime.now() - timedelta(minutes=45),
            "temperature_c": 22.3,
            "temperature_status": "normal",
            "humidity_pct": 55,
            "humidity_status": "normal",
            "heat_stress_index": "normal",
            "feeder_status": "full",
            "waterer_status": "full"
        },
        {
            "timestamp": datetime.now() - timedelta(minutes=30),
            "temperature_c": 28.5,
            "temperature_status": "warning",
            "humidity_pct": 72,
            "humidity_status": "normal",
            "heat_stress_index": "warning",
            "feeder_status": "full",
            "waterer_status": "full"
        },
        {
            "timestamp": datetime.now() - timedelta(minutes=15),
            "temperature_c": 35.2,
            "temperature_status": "critical",
            "humidity_pct": 85,
            "humidity_status": "critical",
            "heat_stress_index": "critical",
            "feeder_status": "low",
            "waterer_status": "empty"
        }
    ]

    for reading in mock_readings:
        insert_sensor_reading(reading)

    print(f"Inserted {len(mock_readings)} mock readings")


if __name__ == "__main__":
    """
    Run this script directly to set up database and insert test data.

    Usage:
        python db_utils.py
    """
    print("Setting up database...")

    try:
        setup_database()
        insert_mock_data()

        # Test query
        latest = get_latest_sensor_reading()
        print(f"\nLatest sensor reading:")
        print(f"   Temperature: {latest['temperature_c']}C [{latest['temperature_status']}]")
        print(f"   Humidity: {latest['humidity_pct']}% [{latest['humidity_status']}]")
        print(f"   Heat stress: {latest['heat_stress_index']}")

    except Exception as e:
        print(f"Database error: {e}")
        print("\nMake sure PostgreSQL is running and .env has correct DB credentials!")
