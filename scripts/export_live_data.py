"""
Export Live Deployment Data
Exports event_log + sensor_readings + reviews to CSV for paper analysis.
"""

import sys, os, csv, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.db_utils import get_db_connection, release_db_connection
from psycopg2.extras import RealDictCursor

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "exports")


def ensure_output_dir():
    os.makedirs(OUTPUT_DIR, exist_ok=True)


def write_csv(filename, rows):
    if not rows:
        print(f"  {filename}: no data")
        return
    path = os.path.join(OUTPUT_DIR, filename)
    keys = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            cleaned = {}
            for k, v in row.items():
                if isinstance(v, (dict, list)):
                    cleaned[k] = json.dumps(v, default=str)
                else:
                    cleaned[k] = v
            writer.writerow(cleaned)
    print(f"  {filename}: {len(rows)} rows -> {path}")


def export_queries():
    """All llm_response events with enhanced logging fields."""
    conn = get_db_connection()
    try:
        cur = conn.cursor(cursor_factory=RealDictCursor)
        cur.execute("""
            SELECT id, timestamp, severity, user_query, llm_response,
                   sensor_snapshot, sensor_context_filtered, sources,
                   routing_mode, routing_decision, prompt_template, response_time_ms
            FROM event_log
            WHERE event_type = 'llm_response'
            ORDER BY id
        """)
        rows = [dict(r) for r in cur.fetchall()]
        cur.close()
        write_csv("live_queries.csv", rows)
        return rows
    finally:
        release_db_connection(conn)


def export_reviews():
    """All llm_response events joined with reviews."""
    conn = get_db_connection()
    try:
        cur = conn.cursor(cursor_factory=RealDictCursor)
        cur.execute("""
            SELECT e.id, e.timestamp, e.severity, e.user_query, e.llm_response,
                   e.sensor_context_filtered, e.routing_mode, e.routing_decision,
                   e.prompt_template, e.response_time_ms,
                   r.is_good, r.routing_correct, r.notes AS review_notes, r.reviewed_at
            FROM event_log e
            LEFT JOIN response_reviews r ON r.event_id = e.id
            WHERE e.event_type = 'llm_response'
            ORDER BY e.id
        """)
        rows = [dict(r) for r in cur.fetchall()]
        cur.close()
        write_csv("live_reviews.csv", rows)
        return rows
    finally:
        release_db_connection(conn)


def export_alerts():
    """All sensor_alert events."""
    conn = get_db_connection()
    try:
        cur = conn.cursor(cursor_factory=RealDictCursor)
        cur.execute("""
            SELECT id, timestamp, severity, sensor_snapshot, sensor_context_filtered
            FROM event_log
            WHERE event_type = 'sensor_alert'
            ORDER BY id
        """)
        rows = [dict(r) for r in cur.fetchall()]
        cur.close()
        write_csv("live_alerts.csv", rows)
        return rows
    finally:
        release_db_connection(conn)


def export_sensor_readings():
    """Raw sensor readings from deployment period."""
    conn = get_db_connection()
    try:
        cur = conn.cursor(cursor_factory=RealDictCursor)
        cur.execute("SELECT * FROM sensor_readings_colson ORDER BY id")
        rows = [dict(r) for r in cur.fetchall()]
        cur.close()
        write_csv("live_sensor_readings.csv", rows)
        return rows
    finally:
        release_db_connection(conn)


if __name__ == "__main__":
    ensure_output_dir()
    print("Exporting live deployment data...")
    export_queries()
    export_reviews()
    export_alerts()
    export_sensor_readings()
    print(f"\nDone. Files in {OUTPUT_DIR}/")
