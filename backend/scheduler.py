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
from apscheduler.triggers.cron import CronTrigger

from backend.db_utils import (
    get_latest_sensor_reading, insert_event, insert_sensor_reading,
    insert_cv_count, insert_risk_snapshot,
)
from backend.sensor_filter import (
    get_sensor_context, get_critical_alerts, is_reading_stale,
    heat_is_critical, mold_is_critical,
)
from backend.rag_functions import answer_query
from backend.egg_reconciliation import reconcile_yesterday

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

    # Build a dedup key from which sensors are critical — NOT from their values.
    # Using values (e.g. "36.2°C") caused a new alert every tick as readings drifted.
    _CRITICAL_FIELDS = ["temperature_status", "humidity_status", "h2s_level"]
    _EMPTY_FIELDS = ["feeder_status", "waterer_status"]
    active = (
        [f for f in _CRITICAL_FIELDS if reading.get(f) == "critical"] +
        [f for f in _EMPTY_FIELDS if reading.get(f) == "empty"]
    )
    if heat_is_critical(reading.get("heat_risk_level")):
        active.append("heat_risk_level")
    if mold_is_critical(reading.get("mold_risk_level")):
        active.append("mold_risk_level")
    alert_key = "|".join(sorted(active))

    if alert_key == _last_alert_type:
        # Same alerts as last cycle — don't spam
        return

    # Determine severity
    temp_critical = reading.get("temperature_status") == "critical"
    heat_critical = heat_is_critical(reading.get("heat_risk_level"))
    h2s_critical = reading.get("h2s_level") == "critical"
    mold_critical = mold_is_critical(reading.get("mold_risk_level"))
    severity = "critical" if (temp_critical or heat_critical or h2s_critical or mold_critical) else "warning"

    sensor_context = get_sensor_context(reading)
    logger.warning(f"Scheduler alert [{severity}]: {'; '.join(critical_alerts)}")

    inserted = False

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
                sensor_data=reading,
            )

            insert_event(
                event_type="sensor_alert",
                severity=severity,
                user_query=query,
                llm_response=result["answer"],
                sensor_snapshot=reading,
                sensor_context_filtered=result["sensor_context"],
                sources=result.get("sources"),
            )
            inserted = True
        except Exception as e:
            logger.error(f"Scheduler: RAG pipeline failed: {e}")

    # Fallback: log without RAG advice (RAG not ready or failed)
    if not inserted:
        try:
            insert_event(
                event_type="sensor_alert",
                severity=severity,
                sensor_snapshot=reading,
                sensor_context_filtered=sensor_context,
            )
            inserted = True
        except Exception as e:
            logger.error(f"Scheduler: fallback event insert failed: {e}")

    # Only update dedup key after a successful insert.
    # If we set it before inserting and the insert fails, future alerts for the
    # same condition are silently suppressed until conditions normalize.
    if inserted:
        _last_alert_type = alert_key
    else:
        logger.error("Scheduler: alert could not be logged to event_log — check DB connection and that setup_database() succeeded")


# ---------------------------------------------------------------------------
# Simulation mode — inserts a fake reading every 60 s
# Enabled by SIMULATION_MODE=true in .env  (switch off → set to false + restart)
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Continuous-drift simulation state
# Values drift smoothly toward time-of-day targets; discrete fields are
# derived from numeric values so the dashboard shows in-between states.
# ---------------------------------------------------------------------------

def _temp_target(hour: int) -> float:
    """Baseline temperature target for time of day (sine-ish curve)."""
    # Peaks ~14:00, trough ~04:00
    import math
    return 22 + 10 * math.sin(math.pi * (hour - 4) / 12) if 4 <= hour <= 16 else \
           22 + 10 * math.sin(math.pi * (hour - 4) / 12)


def _humidity_target(temp: float) -> float:
    """Humidity rises with temperature (inverse relationship, simplified)."""
    return max(40, min(90, 75 - (temp - 22) * 0.8))


def _classify_temp(t: float):
    if t >= 35:
        return "critical"
    if t >= 28 or t <= 10:
        return "warning"
    return "normal"


def _classify_humidity(h: float):
    if h >= 85 or h < 30:
        return "critical"
    if h >= 75 or h < 40:
        return "warning"
    return "normal"


_STATUS_TO_HEAT_LEVEL = {
    "normal": "LOW - Monitor",
    "warning": "MEDIUM - Elevated risk - Prepare to act",
    "critical": "HIGH - Take action now",
}
_STATUS_TO_MOLD_LEVEL = {
    "normal": "low",
    "warning": "medium",
    "critical": "high",
}


def _sim_heat_status(t: float, h: float) -> str:
    hi = t + 0.33 * h - 4  # simplified heat index
    if hi >= 40 or t >= 35:
        return "critical"
    if hi >= 30 or t >= 28:
        return "warning"
    return "normal"


def _sim_thi(t: float, h: float) -> float:
    """Coarse THI approximation for simulation only."""
    return round(0.8 * t + (h / 100.0) * (t - 14.4) + 46.4, 1)


def _classify_h2s(ppm: float):
    if ppm >= 10:
        return "critical"
    if ppm >= 5:
        return "warning"
    return "normal"


def _classify_mold(score: float):
    if score >= 70:
        return "critical"
    if score >= 40:
        return "warning"
    return "normal"


def _feeder_status(pct: float):
    if pct < 15:
        return "empty"
    if pct < 35:
        return "low"
    return "full"


def _waterer_status(pct: float):
    if pct < 15:
        return "empty"
    if pct < 35:
        return "low"
    return "full"


def _drift(current: float, target: float, speed: float, noise: float) -> float:
    """Move current toward target by speed fraction, add noise."""
    return current + (target - current) * speed + random.gauss(0, noise)


# Persistent sim state — initialised on first use
_sim_state: dict | None = None
_sim_event_ticks = 0   # >0 means an event is active


def _init_sim_state():
    hour = datetime.now().hour
    t = _temp_target(hour) + random.gauss(0, 2)
    h = _humidity_target(t) + random.gauss(0, 3)
    return {
        "temperature_c": t,
        "humidity_pct": h,
        "feeder_pct": random.uniform(60, 95),
        "waterer_pct": random.uniform(60, 95),
        "h2s_ppm": random.uniform(0, 2),
        "mold_risk_score": random.uniform(10, 30),
        "chickens_inside": random.randint(8, 12),
        "egg_count": random.randint(0, 5),
        "door_open": False,
        "ventilation_on": False,
    }


def _sim_insert_reading():
    """
    Simulation mode: drift the continuous sensor state and insert one reading.
    Values evolve smoothly; status strings are derived from numeric values.
    Occasional short events (h2s spike, resource drain, heat wave) add variety.
    """
    global _sim_state, _sim_event_ticks

    if _sim_state is None:
        _sim_state = _init_sim_state()

    hour = datetime.now().hour
    s = _sim_state

    # ---- determine targets for this tick ----
    t_target = _temp_target(hour)
    h_target = _humidity_target(s["temperature_c"])

    # Random short events (5% chance each tick, lasts ~3 ticks = ~3 min)
    if _sim_event_ticks <= 0 and random.random() < 0.05:
        event = random.choice(["h2s_spike", "heat_wave", "resource_drain", "cold_snap"])
        logger.debug(f"Sim event triggered: {event}")
        if event == "h2s_spike":
            s["h2s_ppm"] = random.uniform(12, 22)
        elif event == "heat_wave":
            t_target = random.uniform(36, 39)
        elif event == "resource_drain":
            s["feeder_pct"] = max(5, s["feeder_pct"] - random.uniform(30, 50))
            s["waterer_pct"] = max(5, s["waterer_pct"] - random.uniform(20, 40))
        elif event == "cold_snap":
            t_target = random.uniform(5, 12)
        _sim_event_ticks = random.randint(2, 4)
    elif _sim_event_ticks > 0:
        _sim_event_ticks -= 1

    # ---- drift numeric values toward targets ----
    s["temperature_c"] = _drift(s["temperature_c"], t_target, speed=0.15, noise=0.4)
    s["humidity_pct"] = _drift(s["humidity_pct"], h_target, speed=0.12, noise=1.0)
    s["h2s_ppm"] = max(0, _drift(s["h2s_ppm"], 1.0, speed=0.20, noise=0.3))
    s["mold_risk_score"] = max(0, min(100,
        _drift(s["mold_risk_score"],
               20 + (s["humidity_pct"] - 55) * 0.8,
               speed=0.08, noise=1.5)))

    # Resources deplete slowly, reset if refilled (probabilistic refill)
    s["feeder_pct"] = max(0, s["feeder_pct"] - random.uniform(0.1, 0.5))
    s["waterer_pct"] = max(0, s["waterer_pct"] - random.uniform(0.1, 0.6))
    if s["feeder_pct"] < 10 and random.random() < 0.15:
        s["feeder_pct"] = random.uniform(80, 100)   # refilled
    if s["waterer_pct"] < 10 and random.random() < 0.15:
        s["waterer_pct"] = random.uniform(80, 100)

    # Chickens drift based on time (more inside at night)
    inside_target = 12 if (hour >= 20 or hour < 6) else random.uniform(6, 10)
    s["chickens_inside"] = int(round(max(0, min(14,
        _drift(s["chickens_inside"], inside_target, speed=0.1, noise=0.5)))))

    # Egg count: slow accumulation during day, reset chance at end of day
    if 8 <= hour <= 14 and random.random() < 0.3:
        s["egg_count"] = min(s["egg_count"] + 1, 15)
    elif hour >= 20 and random.random() < 0.1:
        s["egg_count"] = 0   # collected

    # Door and ventilation logic
    temp = s["temperature_c"]
    s["ventilation_on"] = temp >= 28 or s["h2s_ppm"] >= 5
    s["door_open"] = 6 <= hour <= 20 and temp < 35

    # ---- build reading with derived status fields ----
    # cv data (chickens, eggs) goes to cv_counts_colson via insert_cv_count
    # heat/mold risk now goes to risk_snapshots via insert_risk_snapshot
    reading = {
        "temperature_c": round(s["temperature_c"], 2),
        "temperature_status": _classify_temp(s["temperature_c"]),
        "humidity_pct": round(s["humidity_pct"], 2),
        "humidity_status": _classify_humidity(s["humidity_pct"]),
        "feeder_pct": round(s["feeder_pct"], 1),
        "feeder_status": _feeder_status(s["feeder_pct"]),
        "waterer_pct": round(s["waterer_pct"], 1),
        "waterer_status": _waterer_status(s["waterer_pct"]),
        "h2s_ppm": round(s["h2s_ppm"], 2),
        "h2s_level": _classify_h2s(s["h2s_ppm"]),
        "door_open": s["door_open"],
        "ventilation_on": s["ventilation_on"],
    }

    # Derived risk snapshot (simulation-grade approximation)
    heat_status = _sim_heat_status(s["temperature_c"], s["humidity_pct"])
    mold_status = _classify_mold(s["mold_risk_score"])
    thi = _sim_thi(s["temperature_c"], s["humidity_pct"])
    heat_score_map = {"normal": 20, "warning": 60, "critical": 90}

    try:
        insert_sensor_reading(reading)
        insert_cv_count(s["chickens_inside"], s["egg_count"])
        insert_risk_snapshot(
            heat_risk_score=heat_score_map[heat_status],
            heat_risk_level=_STATUS_TO_HEAT_LEVEL[heat_status],
            thi_current=thi,
            mold_risk_score=round(s["mold_risk_score"]),
            mold_risk_level=_STATUS_TO_MOLD_LEVEL[mold_status],
            mold_favourable_for_growth=s["humidity_pct"] > 80,
            contributing_factors=f"sim: temp={reading['temperature_c']}C, rh={reading['humidity_pct']}%",
        )
        logger.debug(
            f"Sim: T={reading['temperature_c']}°C H={reading['humidity_pct']}% "
            f"feeder={reading['feeder_pct']}% h2s={reading['h2s_ppm']}ppm "
            f"[{reading['temperature_status']}]"
        )
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

    # Daily egg reconciliation at 23:55 local time.
    # Preserves manual edits; uses positive-delta sum over cv_counts_colson.
    def _safe_reconcile():
        try:
            reconcile_yesterday()
        except Exception as e:
            logger.error(f"Daily egg reconciliation failed: {e}")

    _scheduler.add_job(
        _safe_reconcile,
        trigger=CronTrigger(hour=23, minute=55),
        id="egg_reconcile_daily",
        name="Daily egg calendar reconciliation",
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
