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
import random
from datetime import datetime
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.interval import IntervalTrigger

from db_utils import get_latest_sensor_reading, insert_event, insert_sensor_reading
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
    h2s_critical = reading.get("h2s_level") == "critical"
    mold_critical = reading.get("mold_risk_status") == "critical"
    severity = "critical" if (temp_critical or stress_critical or h2s_critical or mold_critical) else "warning"

    sensor_context = get_sensor_context(reading)
    logger.warning(f"Scheduler alert [{severity}]: {'; '.join(critical_alerts)}")

    # Run RAG to generate actionable advice
    if _vectordb is not None and _bm25_retriever is not None:
        query = "ALERT: " + "; ".join(critical_alerts) + ". What must I do right now to protect my flock?"

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
# Simulation mode — inserts a fake reading every 60 s
# Enabled by SIMULATION_MODE=true in .env  (switch off → set to false + restart)
# ---------------------------------------------------------------------------

_SIM_SCENARIOS = {
    "normal": {
        "temperature_c": lambda: random.uniform(20, 24),
        "temperature_status": "normal",
        "humidity_pct": lambda: random.uniform(50, 70),
        "humidity_status": "normal",
        "heat_stress_index": "normal",
        "feeder_status": "full",
        "waterer_status": "full",
        "feeder_pct": lambda: random.uniform(70, 100),
        "waterer_pct": lambda: random.uniform(70, 100),
        "chickens_inside": lambda: random.randint(8, 12),
        "egg_count": lambda: random.randint(0, 5),
        "h2s_ppm": lambda: random.uniform(0, 2),
        "h2s_level": "normal",
        "mold_risk_score": lambda: random.uniform(0, 20),
        "mold_risk_status": "normal",
        "door_open": False,
        "ventilation_on": False,
    },
    "hot_day": {
        "temperature_c": lambda: random.uniform(28, 32),
        "temperature_status": "warning",
        "humidity_pct": lambda: random.uniform(65, 80),
        "humidity_status": "warning",
        "heat_stress_index": "warning",
        "feeder_status": lambda: random.choice(["full", "full", "low"]),
        "waterer_status": lambda: random.choice(["full", "low"]),
        "feeder_pct": lambda: random.uniform(40, 80),
        "waterer_pct": lambda: random.uniform(30, 70),
        "chickens_inside": lambda: random.randint(5, 10),
        "egg_count": lambda: random.randint(0, 3),
        "h2s_ppm": lambda: random.uniform(1, 5),
        "h2s_level": "normal",
        "mold_risk_score": lambda: random.uniform(20, 50),
        "mold_risk_status": lambda: random.choice(["normal", "warning"]),
        "door_open": True,
        "ventilation_on": True,
    },
    "critical": {
        "temperature_c": lambda: random.uniform(35, 38),
        "temperature_status": "critical",
        "humidity_pct": lambda: random.uniform(80, 90),
        "humidity_status": "critical",
        "heat_stress_index": "critical",
        "feeder_status": lambda: random.choice(["low", "empty"]),
        "waterer_status": lambda: random.choice(["low", "empty"]),
        "feeder_pct": lambda: random.uniform(5, 25),
        "waterer_pct": lambda: random.uniform(5, 20),
        "chickens_inside": lambda: random.randint(2, 8),
        "egg_count": 0,
        "h2s_ppm": lambda: random.uniform(10, 25),
        "h2s_level": lambda: random.choice(["warning", "critical"]),
        "mold_risk_score": lambda: random.uniform(60, 95),
        "mold_risk_status": "critical",
        "door_open": lambda: random.choice([True, False]),
        "ventilation_on": False,
    },
    "cold_night": {
        "temperature_c": lambda: random.uniform(8, 14),
        "temperature_status": "warning",
        "humidity_pct": lambda: random.uniform(60, 75),
        "humidity_status": "normal",
        "heat_stress_index": "normal",
        "feeder_status": "full",
        "waterer_status": "full",
        "feeder_pct": lambda: random.uniform(60, 100),
        "waterer_pct": lambda: random.uniform(60, 100),
        "chickens_inside": lambda: random.randint(10, 14),
        "egg_count": lambda: random.randint(0, 8),
        "h2s_ppm": lambda: random.uniform(0, 3),
        "h2s_level": "normal",
        "mold_risk_score": lambda: random.uniform(30, 60),
        "mold_risk_status": lambda: random.choice(["normal", "warning"]),
        "door_open": False,
        "ventilation_on": lambda: random.choice([True, False]),
    },
    "resource_low": {
        "temperature_c": lambda: random.uniform(20, 24),
        "temperature_status": "normal",
        "humidity_pct": lambda: random.uniform(50, 65),
        "humidity_status": "normal",
        "heat_stress_index": "normal",
        "feeder_status": "low",
        "waterer_status": "low",
        "feeder_pct": lambda: random.uniform(5, 20),
        "waterer_pct": lambda: random.uniform(5, 20),
        "chickens_inside": lambda: random.randint(8, 12),
        "egg_count": lambda: random.randint(0, 4),
        "h2s_ppm": lambda: random.uniform(0, 3),
        "h2s_level": "normal",
        "mold_risk_score": lambda: random.uniform(10, 35),
        "mold_risk_status": "normal",
        "door_open": False,
        "ventilation_on": False,
    },
}


def _resolve(val):
    return val() if callable(val) else val


def _sim_insert_reading():
    """
    Simulation mode: insert one synthetic sensor reading.
    Scenario follows a realistic daily temperature pattern.
    """
    hour = datetime.now().hour
    if 6 <= hour < 10:
        scenario = "normal"
    elif 10 <= hour < 14:
        scenario = "hot_day"
    elif 14 <= hour < 16:
        scenario = random.choice(["hot_day", "critical"])
    elif 16 <= hour < 20:
        scenario = "hot_day"
    elif 20 <= hour < 22:
        scenario = "normal"
    else:
        scenario = "cold_night"
    r = random.random()
    if r < 0.20:
        scenario = "critical"
    elif r < 0.30:
        scenario = "resource_low"

    reading = {k: _resolve(v) for k, v in _SIM_SCENARIOS[scenario].items()}
    try:
        insert_sensor_reading(reading)
        logger.debug(f"Sim: inserted '{scenario}' reading")
    except Exception as e:
        logger.error(f"Sim: failed to insert reading: {e}")


# ---------------------------------------------------------------------------
# Scheduler lifecycle (called from app.py lifespan)
# ---------------------------------------------------------------------------

_scheduler = None


def start_scheduler(interval_seconds: int = 60, vectordb=None, bm25_retriever=None,
                    simulation_mode: bool = False):
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
    if simulation_mode:
        _scheduler.add_job(
            _sim_insert_reading,
            trigger=IntervalTrigger(seconds=60),
            id="sim_sensor",
            name="Simulation: insert fake reading every 60 s",
            replace_existing=True,
        )

    _scheduler.start()
    mode = "SIMULATION" if simulation_mode else "live"
    logger.info(f"Scheduler started ({mode}, check={interval_seconds}s, rag={'yes' if vectordb else 'no'})")


def stop_scheduler():
    """Shut down the background scheduler gracefully."""
    global _scheduler
    if _scheduler is not None:
        _scheduler.shutdown(wait=False)
        _scheduler = None
        logger.info("Scheduler stopped")
