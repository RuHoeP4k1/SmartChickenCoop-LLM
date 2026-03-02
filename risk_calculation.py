from __future__ import annotations
import math
from typing import Any, Dict, List

from datetime import datetime
from typing import Any, Dict, List, Optional

#=============================================================================
# ADAPTER LAYER — translating raw data to risk inputs
#=============================================================================

def _parse_timestamp(value: Any) -> Optional[datetime]:
    if isinstance(value, datetime):
        return value
    if isinstance(value, str):
        normalized = value.replace("Z", "+00:00")
        try:
            return datetime.fromisoformat(normalized)
        except ValueError:
            return None
    return None

def build_heat_risk_inputs_from_recent_readings(
    recent_readings: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Derive every `compute_heat_risk` input from rows returned by
    `db_utils.get_recent_readings()`.
    """
    if not recent_readings:
        raise ValueError("recent_readings is empty")

 # 1 hour window at 10-minute cadence -> 6 most recent intervals
    one_hour_window = recent_readings[:6]

    temps = [float(r["temperature_c"]) for r in recent_readings if r.get("temperature_c") is not None] #TODO: handle missing values better
    rhs = [float(r["humidity_pct"]) for r in recent_readings if r.get("humidity_pct") is not None] #TODO: handle missing values better

    if not temps:
        raise ValueError("No temperature values found in recent_readings")
    if not rhs:
        raise ValueError("No humidity values found in recent_readings")

    thi_series: List[Dict[str, Any]] = []
    for row in recent_readings:
        t = row.get("temperature_c")
        rh = row.get("humidity_pct")
        if t is None or rh is None:
            continue
        t = float(t)
        rh = float(rh)
        twb = wet_bulb_temperature_c(t, rh)
        thi = 0.85 * t + 0.15 * twb
        thi_series.append(
            {
                "timestamp": _parse_timestamp(row.get("timestamp")),
                "thi": thi,
                "row": row,
            }
        )

    latest = recent_readings[0]
    feeder_status = str(latest.get("feeder_status", "")).lower()
    waterer_status = str(latest.get("waterer_status", "")).lower()

    feeder_pct = latest.get("feeder_pct")
    waterer_pct = latest.get("waterer_pct")

    feed_intake = "Normal"
    if feeder_status in {"low", "empty"}:
        feed_intake = "Reduced"
    elif feeder_pct is not None and float(feeder_pct) < 35:
        feed_intake = "Reduced"

    water_intake = "Normal"
    if waterer_status in {"low", "empty"}:
        water_intake = "High"
    elif waterer_pct is not None and float(waterer_pct) < 35:
        water_intake = "High"

    high_thi_threshold = 27.0
    high_streak = []
    for point in thi_series:
        if point["thi"] >= high_thi_threshold:
            high_streak.append(point)
        else:
            break

    sampling_interval = 10.0 # in minutes
    high_thi_streak_minutes = 0
    if high_streak:
        streak_ts = [p["timestamp"] for p in high_streak if p["timestamp"] is not None]
        if len(streak_ts) >= 2:
            newest = max(streak_ts)
            oldest = min(streak_ts)
            high_thi_streak_minutes = int((newest - oldest).total_seconds() / 60.0 + sampling_interval)
        else:
            high_thi_streak_minutes = int(sampling_interval)

    thi_slope_per_hour: Optional[float] = None
    thi_with_ts = [p for p in thi_series if p["timestamp"] is not None]
    if len(thi_with_ts) >= 2:
        thi_with_ts.sort(key=lambda x: x["timestamp"])
        first = thi_with_ts[0]
        last = thi_with_ts[-1]
        delta_hours = (last["timestamp"] - first["timestamp"]).total_seconds() / 3600.0
        if delta_hours > 0:
            thi_slope_per_hour = (last["thi"] - first["thi"]) / delta_hours

    data_coverage_last_hour: Optional[float] = None
    if thi_with_ts:
        newest_ts = max(p["timestamp"] for p in thi_with_ts)
        since = newest_ts.timestamp() - 3600.0
        count_last_hour = sum(1 for p in thi_with_ts if p["timestamp"].timestamp() >= since)
        expected_points = max(1, int(60.0 / max(1.0, sampling_interval)))
        data_coverage_last_hour = min(1.0, count_last_hour / expected_points)

    return {
        "temp_db_mean": sum(temps) / len(temps),
        "temp_db_max": max(temps),
        "rh_percent_mean": sum(rhs) / len(rhs),
        "high_thi_streak_minutes": high_thi_streak_minutes,
        "feed_intake": feed_intake,
        "water_intake": water_intake,
        "thi_slope_per_hour": thi_slope_per_hour,
        "data_coverage_last_hour": data_coverage_last_hour,
        "sensor_count_temp": 1,
    }







#=============================================================================
#WET BULP CALCULATION
#=============================================================================
def wet_bulb_temperature_c(t_db_c: float, rh_percent: float) -> float:
    """
    Approximate wet-bulb temperature (°C) from dry-bulb temperature (°C)
    and relative humidity (%) using Stull (2011) approximation.

    Valid for typical ambient conditions (roughly 0–50°C, 5–99% RH).
    Good enough for control/early warning use cases.
    """
    rh = max(1.0, min(rh_percent, 99.0))  # keep in a safe range
    t = t_db_c

    twb = (
        t * math.atan(0.151977 * math.sqrt(rh + 8.313659))
        + math.atan(t + rh)
        - math.atan(rh - 1.676331)
        + 0.00391838 * (rh ** 1.5) * math.atan(0.023101 * rh)
        - 4.686035
    )
    return twb
# =============================================================================
# SAFETY RULES — general guidelines for all prompts
# =============================================================================
from __future__ import annotations
import math
from typing import Any, Dict, List, Optional


def wet_bulb_temperature_c(t_db_c: float, rh_percent: float) -> float:
    rh = max(1.0, min(rh_percent, 99.0))
    t = t_db_c
    return (
        t * math.atan(0.151977 * math.sqrt(rh + 8.313659))
        + math.atan(t + rh)
        - math.atan(rh - 1.676331)
        + 0.00391838 * (rh ** 1.5) * math.atan(0.023101 * rh)
        - 4.686035
    )


def compute_heat_risk(
    temp_db_mean: float,
    temp_db_max: float,
    rh_percent_mean: float,  # 0..100
    high_thi_streak_minutes: int,  # e.g. minutes above THI threshold
    feed_intake: str,
    water_intake: str,
    thi_slope_per_hour: Optional[float] = None,  # optional early warning
) -> Dict[str, Any]:
    score = 0.0
    contributing_factors: List[str] = []

    twb_mean = wet_bulb_temperature_c(temp_db_mean, rh_percent_mean)
    thi_mean = 0.85 * temp_db_mean + 0.15 * twb_mean

    twb_max = wet_bulb_temperature_c(temp_db_max, rh_percent_mean)
    thi_max = 0.85 * temp_db_max + 0.15 * twb_max

    # --- Base risk from THI (make high end steeper)
    if thi_mean < 19:
        score = 0.0
        contributing_factors.append("THI within safe range")
    elif thi_mean < 22:
        score = 0.2
        contributing_factors.append("THI elevated")
    elif thi_mean < 25:
        score = 0.5
        contributing_factors.append("THI moderately high")
    elif thi_mean < 29:
        score = 0.75
        contributing_factors.append("THI high")
    elif thi_mean < 31:
        score = 0.85
        contributing_factors.append("THI very high")
    elif thi_mean < 33:
        score = 0.95
        contributing_factors.append("THI critical")
    else:
        score = 1.0
        contributing_factors.append("THI extreme")

    # --- Peak risk (hotspots)
    if thi_max >= 33:
        score += 0.1
        contributing_factors.append("Critical peak conditions (hotspot)")

    # --- Prolonged exposure
    if high_thi_streak_minutes >= 30:
        score += 0.1
        contributing_factors.append("Sustained exposure (≥30 min)")

    # --- Resource tracking signals
    if feed_intake in {"Reduced", "Low"}:
        score += 0.1
        contributing_factors.append("Reduced feed intake")
    if water_intake in {"Increased", "High"}:
        score += 0.1
        contributing_factors.append("Increased water uptake")

    # --- Trend-based early warning (optional)
    if thi_slope_per_hour is not None and thi_slope_per_hour >= 2.0:
        score += 0.05
        contributing_factors.append("THI rising quickly")

    score = max(0.0, min(score, 1.0))

    # --- Risk level thresholds (keep simple)
    if score < 0.5:
        level = "LOW"
    elif score < 0.75:
        level = "MEDIUM"
    else:
        level = "HIGH"

    # --- Horizon (urgency)
    # Interpreting as: time before conditions likely become harmful if nothing changes
    time_horizon = 240
    if thi_mean >= 33:
        time_horizon = 10
    elif thi_mean >= 31:
        time_horizon = 30
    elif thi_mean >= 29:
        time_horizon = 60
    elif thi_mean >= 25:
        time_horizon = 120

    # If already sustained, shorten horizon ==> more urgent to act because animals are already stressed
    if high_thi_streak_minutes >= 60:
        time_horizon = min(time_horizon, 30)

    # If THI rising fast, shorten horizon ==> early warning that conditions may deteriorate quickly
    if thi_slope_per_hour is not None and thi_slope_per_hour >= 2.0:
        time_horizon = max(10, time_horizon // 2)

    return {
        "event_type": "HEAT_RISK",
        "risk_score": round(score * 100, 1),
        "risk_level": level,
        "time_horizon_minutes": int(time_horizon),
        "confidence": round(confidence * 100, 0),  # percent
        "contributing_factors": contributing_factors,
        "thi_mean": round(thi_mean, 2),
        "thi_max": round(thi_max, 2),
    }

def decide_ventilation_action(
    risk: Dict[str, Any],
    current_fan_percent: int,
    min_fan_percent: int = 15,
    max_fan_percent: int = 100,
) -> Dict[str, Any]:
    """Zet risk-output om naar een concrete ventilatie-actie.

    Geeft een payload terug die je rechtstreeks kan loggen, publishen naar MQTT,
    of doorgeven aan je RAG-laag voor uitleg aan de gebruiker.
    """

    risk_level = risk["risk_level"]
    risk_score = float(risk["risk_score"])
    time_horizon = int(risk["time_horizon_minutes"])

    if risk_level == "LOW":
        target_fan = max(min_fan_percent, 20)
        action = "MAINTAIN"
    elif risk_level == "MEDIUM":
        target_fan = max(min_fan_percent, 45)
        action = "INCREASE"
    else:  # HIGH
        if risk_score >= 0.9 or time_horizon <= 10:
            target_fan = max_fan_percent
            action = "EMERGENCY_MAX"
        else:
            target_fan = min(max_fan_percent, 75)
            action = "INCREASE_STRONGLY"

    delta = target_fan - current_fan_percent

    return {
        "controller": "VENTILATION",
        "action": action,
        "current_fan_percent": current_fan_percent,
        "target_fan_percent": target_fan,
        "delta_percent": delta,
        "reason": {
            "risk_level": risk_level,
            "risk_score": risk_score,
            "time_horizon_minutes": time_horizon,
            "contributing_factors": risk["contributing_factors"],
        },
    }


def build_rag_context(risk: Dict[str, Any], ventilation_action: Dict[str, Any]) -> Dict[str, Any]:
    """Combineer detectie + actuatorbeslissing voor downstream RAG-advies."""

    return {
        "event_type": "HEAT_RISK_CONTROL",
        "risk": risk,
        "ventilation": ventilation_action,
        "operator_prompt": (
            "Leg in duidelijke taal uit waarom deze ventilatie-actie wordt uitgevoerd, "
            "welke bijkomende acties de pluimveehouder nu best doet (water, schaduw, dichtheid), "
            "en welke check over 10 minuten nodig is."
        ),
    }
