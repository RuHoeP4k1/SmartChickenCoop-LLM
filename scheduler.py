"""
Background Scheduler for ChickenCare AI

Periodically checks the latest sensor reading in the DB.
If something critical is detected, it runs the RAG pipeline to generate
actionable advice and logs everything to event_log so the frontend can
show it to the user.

Who decides what's critical? The sensor team — they write the status
values (normal/warning/critical) into sensor_readings from the Pi.
This scheduler reads those, and when critical, asks the RAG for advice.

Designed to run inside the FastAPI app (not standalone).
"""

import logging
from datetime import datetime
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.interval import IntervalTrigger

from db_utils import get_latest_sensor_reading, insert_event
from sensor_filter import get_sensor_context, get_critical_alerts, is_reading_stale
from rag_functions import answer_query

logger = logging.getLogger("chickencareai.scheduler")

# Track last alert to avoid spamming the same alert every cycle
_last_alert_type = None

# RAG state — set by start_scheduler from app.py
_vectordb = None
_bm25_retriever = None


def check_sensors():
    """
    Periodic task: read latest sensors, run RAG if critical.
    Called by APScheduler every SCHEDULER_INTERVAL seconds.

    When critical conditions are found:
    1. Builds a query from the critical alerts
    2. Runs the full RAG pipeline (retrieval + LLM) to get actionable advice
    3. Logs query + advice + sensor data + filtered context to event_log
    """
    global _last_alert_type

    try:
        reading = get_latest_sensor_reading()
    except Exception as e:
        logger.error(f"Scheduler: DB read failed: {e}")
        return

    if not reading:
        return

    if is_reading_stale(reading):
        logger.warning("Scheduler: latest sensor reading is stale, skipping")
        return

    critical_alerts = get_critical_alerts(reading)

    if not critical_alerts:
        # Situation cleared — reset dedup tracker
        if _last_alert_type is not None:
            logger.info("Scheduler: conditions back to normal")
            insert_event(
                event_type="conditions_normal",
                severity="info",
                sensor_snapshot=reading,
                sensor_context_filtered="All sensor readings have returned to normal.",
            )
        _last_alert_type = None
        return

    # Build a dedup key from sorted alert types
    alert_key = "|".join(sorted(critical_alerts))

    if alert_key == _last_alert_type:
        # Same alerts as last cycle — don't spam
        return

    _last_alert_type = alert_key

    # Determine severity
    temp_critical = reading.get("temperature_status") == "critical"
    stress_critical = reading.get("heat_stress_index") == "critical"
    severity = "critical" if (temp_critical or stress_critical) else "warning"

    sensor_context = get_sensor_context(reading)
    logger.warning(f"Scheduler alert [{severity}]: {'; '.join(critical_alerts)}")

    # Run RAG to generate actionable advice
    if _vectordb is not None and _bm25_retriever is not None:
        query = "Critical coop conditions detected: " + "; ".join(critical_alerts)

        try:
            result = answer_query(
                query=query,
                vectordb=_vectordb,
                bm25_retriever=_bm25_retriever,
                use_sensors=True,
                use_hybrid=True,
            )

            insert_event(
                event_type="sensor_alert",
                severity=severity,
                user_query=query,
                llm_response=result["answer"],
                sensor_snapshot=reading,
                sensor_context_filtered=result["sensor_context"],
            )
            return
        except Exception as e:
            logger.error(f"Scheduler: RAG pipeline failed: {e}")

    # Fallback: log without RAG advice (RAG not ready or failed)
    insert_event(
        event_type="sensor_alert",
        severity=severity,
        sensor_snapshot=reading,
        sensor_context_filtered=sensor_context,
    )


# ---------------------------------------------------------------------------
# Scheduler lifecycle (called from app.py lifespan)
# ---------------------------------------------------------------------------

_scheduler = None


def start_scheduler(interval_seconds: int = 60 * 15, vectordb=None, bm25_retriever=None):
    """
    Start the background scheduler.

    Args:
        interval_seconds: Check interval (default 15 min)
        vectordb: Chroma vector store from app state
        bm25_retriever: BM25 retriever from app state
    """
    global _scheduler, _vectordb, _bm25_retriever

    _vectordb = vectordb
    _bm25_retriever = bm25_retriever

    if _scheduler is not None:
        logger.warning("Scheduler already running")
        return

    _scheduler = BackgroundScheduler(daemon=True)
    _scheduler.add_job(
        check_sensors,
        trigger=IntervalTrigger(seconds=interval_seconds),
        id="sensor_check",
        name="Periodic sensor check",
        replace_existing=True,
        next_run_time=datetime.now(),  # run once immediately on startup
    )
    _scheduler.start()
    logger.info(f"Scheduler started (interval={interval_seconds}s, rag={'yes' if vectordb else 'no'})")


def stop_scheduler():
    """Shut down the background scheduler gracefully."""
    global _scheduler
    if _scheduler is not None:
        _scheduler.shutdown(wait=False)
        _scheduler = None
        logger.info("Scheduler stopped")
