"""
Clear Tables for Live Deployment
Truncates event_log + response_reviews so live deployment starts clean.
Leaves sensor_readings_colson and cv_counts_colson untouched (real Pi data).
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.db_utils import get_db_connection, release_db_connection, setup_database
from psycopg2.extras import RealDictCursor


_CLEARABLE_TABLES = [
    "event_log",
    "response_reviews",
    "egg_calendar_entries",
    "chore_log",
    "automation_windows",
    "risk_snapshots",
]


def get_row_counts(conn):
    cur = conn.cursor(cursor_factory=RealDictCursor)
    counts = {}
    for table in _CLEARABLE_TABLES:
        try:
            cur.execute(f"SELECT COUNT(*) AS n FROM {table}")
            counts[table] = cur.fetchone()["n"]
        except Exception:
            conn.rollback()
            counts[table] = "(table missing)"
    cur.close()
    return counts


def main():
    # First ensure schema is up to date (creates tables + runs migrations)
    print("Running setup_database() to ensure schema is current...")
    setup_database()

    conn = get_db_connection()
    try:
        print("\nCurrent row counts:")
        counts = get_row_counts(conn)
        for table, n in counts.items():
            print(f"  {table}: {n}")

        total = sum(v for v in counts.values() if isinstance(v, int))
        if total == 0:
            print("\nAll tables already empty. Nothing to do.")
            return

        print(f"\nThis will TRUNCATE {total} rows from: {', '.join(_CLEARABLE_TABLES)}.")
        print("sensor_readings_colson, cv_counts_colson, chore_definitions will NOT be touched.")
        confirm = input("Type 'yes' to proceed: ").strip().lower()
        if confirm != "yes":
            print("Aborted.")
            return

        cur = conn.cursor()
        cur.execute("""
            TRUNCATE TABLE
                response_reviews, event_log,
                egg_calendar_entries, chore_log, automation_windows, risk_snapshots
            RESTART IDENTITY CASCADE
        """)
        conn.commit()
        cur.close()

        print("\nTables truncated. Verifying...")
        counts = get_row_counts(conn)
        for table, n in counts.items():
            print(f"  {table}: {n}")
        print("\nReady for live deployment.")
    finally:
        release_db_connection(conn)


if __name__ == "__main__":
    main()
