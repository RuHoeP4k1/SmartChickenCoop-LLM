"""
Analyze Live Deployment Data
Runs analysis queries on event_log + response_reviews for the paper.
Outputs summary tables and statistics.
"""

import sys, os, csv, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from collections import Counter
from backend.db_utils import get_db_connection, release_db_connection
from psycopg2.extras import RealDictCursor

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "exports")


def query_db(sql):
    conn = get_db_connection()
    try:
        cur = conn.cursor(cursor_factory=RealDictCursor)
        cur.execute(sql)
        rows = [dict(r) for r in cur.fetchall()]
        cur.close()
        return rows
    finally:
        release_db_connection(conn)


def print_table(title, rows, keys=None):
    if not rows:
        print(f"\n{title}: no data\n")
        return
    keys = keys or list(rows[0].keys())
    widths = {k: max(len(str(k)), max(len(str(r.get(k, ""))) for r in rows)) for k in keys}
    header = " | ".join(str(k).ljust(widths[k]) for k in keys)
    print(f"\n{'='*len(header)}")
    print(title)
    print(f"{'='*len(header)}")
    print(header)
    print("-+-".join("-" * widths[k] for k in keys))
    for r in rows:
        print(" | ".join(str(r.get(k, "")).ljust(widths[k]) for k in keys))
    print()


def overview():
    """Basic deployment statistics."""
    rows = query_db("""
        SELECT
            COUNT(*) FILTER (WHERE event_type = 'llm_response') AS total_queries,
            COUNT(*) FILTER (WHERE event_type = 'sensor_alert') AS total_alerts,
            COUNT(*) FILTER (WHERE event_type = 'conditions_normal') AS normal_events,
            MIN(timestamp) AS first_event,
            MAX(timestamp) AS last_event
        FROM event_log
    """)
    print_table("Deployment Overview", rows)


def routing_distribution():
    """How many queries used LLM vs keyword routing, include vs exclude."""
    rows = query_db("""
        SELECT
            COALESCE(routing_mode, 'unknown') AS routing_mode,
            COALESCE(routing_decision, 'unknown') AS routing_decision,
            COUNT(*) AS count
        FROM event_log
        WHERE event_type = 'llm_response'
        GROUP BY routing_mode, routing_decision
        ORDER BY count DESC
    """)
    print_table("Routing Distribution", rows)


def latency_stats():
    """Response time percentiles."""
    rows = query_db("""
        SELECT
            COUNT(*) AS n,
            ROUND(AVG(response_time_ms)) AS avg_ms,
            PERCENTILE_CONT(0.50) WITHIN GROUP (ORDER BY response_time_ms)::int AS p50_ms,
            PERCENTILE_CONT(0.90) WITHIN GROUP (ORDER BY response_time_ms)::int AS p90_ms,
            PERCENTILE_CONT(0.99) WITHIN GROUP (ORDER BY response_time_ms)::int AS p99_ms,
            MIN(response_time_ms) AS min_ms,
            MAX(response_time_ms) AS max_ms
        FROM event_log
        WHERE event_type = 'llm_response' AND response_time_ms IS NOT NULL
    """)
    print_table("Response Latency", rows)


def review_summary():
    """Review coverage and quality breakdown."""
    rows = query_db("""
        SELECT
            COUNT(*) AS total_queries,
            COUNT(r.id) AS reviewed,
            COUNT(*) FILTER (WHERE r.is_good = true) AS good,
            COUNT(*) FILTER (WHERE r.is_good = false) AS bad,
            COUNT(*) FILTER (WHERE r.routing_correct = true) AS routing_correct,
            COUNT(*) FILTER (WHERE r.routing_correct = false) AS routing_wrong
        FROM event_log e
        LEFT JOIN response_reviews r ON r.event_id = e.id
        WHERE e.event_type = 'llm_response'
    """)
    print_table("Review Summary", rows)


def quality_by_routing():
    """Response quality grouped by whether sensors were included."""
    rows = query_db("""
        SELECT
            COALESCE(e.routing_decision, 'unknown') AS routing_decision,
            COUNT(r.id) AS reviewed,
            COUNT(*) FILTER (WHERE r.is_good = true) AS good,
            COUNT(*) FILTER (WHERE r.is_good = false) AS bad,
            CASE WHEN COUNT(r.id) > 0
                THEN ROUND(100.0 * COUNT(*) FILTER (WHERE r.is_good = true) / COUNT(r.id), 1)
                ELSE 0
            END AS pct_good
        FROM event_log e
        LEFT JOIN response_reviews r ON r.event_id = e.id
        WHERE e.event_type = 'llm_response'
        GROUP BY e.routing_decision
        ORDER BY routing_decision
    """)
    print_table("Quality by Routing Decision", rows)


def hourly_usage():
    """Queries per hour of day."""
    rows = query_db("""
        SELECT
            EXTRACT(HOUR FROM timestamp)::int AS hour,
            COUNT(*) AS queries
        FROM event_log
        WHERE event_type = 'llm_response'
        GROUP BY hour
        ORDER BY hour
    """)
    print_table("Usage by Hour", rows)


def severity_distribution():
    """How many queries triggered critical vs info responses."""
    rows = query_db("""
        SELECT severity, COUNT(*) AS count
        FROM event_log
        WHERE event_type = 'llm_response'
        GROUP BY severity
        ORDER BY count DESC
    """)
    print_table("Severity Distribution", rows)


if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("Live Deployment Analysis")
    print("=" * 60)
    overview()
    routing_distribution()
    latency_stats()
    review_summary()
    quality_by_routing()
    hourly_usage()
    severity_distribution()
    print("Done.")
