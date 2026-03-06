from __future__ import annotations
import math
from typing import Any, Dict, List
from enum import Enum
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from pydantic.dataclasses import dataclass

#=============================================================================
# ADAPTER LAYER — translating raw data to risk inputs
#=============================================================================

def _parse_timestamp(value: Any) -> Optional[datetime]:
    """Parse datetime/ISO timestamp and return timezone-aware UTC datetime."""
    if isinstance(value, datetime):
        dt = value
    elif isinstance(value, str):
        normalized = value.strip().replace("Z", "+00:00")
        try:
            dt = datetime.fromisoformat(normalized)
        except ValueError:
            return None
    else:
        return None

    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _sort_readings_newest_first(readings: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Sort readings by timestamp descending.
    Missing/invalid timestamps are treated as oldest.
    """
    oldest_utc = datetime.min.replace(tzinfo=timezone.utc)

    def sort_key(row: Dict[str, Any]) -> tuple[int, datetime]:
        ts = _parse_timestamp(row.get("timestamp"))
        if ts is None:
            return (0, oldest_utc)
        return (1, ts)

    return sorted(readings, key=sort_key, reverse=True)

def build_heat_risk_inputs_from_recent_readings(
    recent_readings: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Derive every `compute_heat_risk` input from rows returned by
    `db_utils.get_recent_readings()`.
    """
    if not recent_readings:
        raise ValueError("recent_readings is empty")

    # 1 hour window at 10-minute cadence -> 6 most recent rows.
    readings_sorted = _sort_readings_newest_first(recent_readings)
    window = readings_sorted[:6]

    temps = [float(r["temperature_c"]) for r in window if r.get("temperature_c") is not None]
    rhs = [float(r["humidity_pct"]) for r in window if r.get("humidity_pct") is not None]

    if not temps:
        raise ValueError("No temperature values found in recent_readings")
    if not rhs:
        raise ValueError("No humidity values found in recent_readings")

    thi_series: List[Dict[str, Any]] = []
    for row in window:
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

    latest = window[0]
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

    high_thi_threshold = 26.0
    sampling_interval = 10.0  # in minutes

    # Count consecutive high-THI points from newest backwards.
    # Stop on first below-threshold or missing temp/RH.
    count_consecutive_high = 0
    for row in window:
        t = row.get("temperature_c")
        rh = row.get("humidity_pct")
        if t is None or rh is None:
            break

        t_val = float(t)
        rh_val = float(rh)
        twb = wet_bulb_temperature_c(t_val, rh_val)
        thi = 0.85 * t_val + 0.15 * twb
        if thi >= high_thi_threshold:
            count_consecutive_high += 1
        else:
            break

    high_thi_streak_minutes = int(count_consecutive_high * sampling_interval)

    thi_slope_per_hour: Optional[float] = None
    thi_with_ts = [p for p in thi_series if p["timestamp"] is not None]
    if len(thi_with_ts) >= 2:
        thi_with_ts.sort(key=lambda x: x["timestamp"])
        first = thi_with_ts[0]
        last = thi_with_ts[-1]
        delta_hours = (last["timestamp"] - first["timestamp"]).total_seconds() / 3600.0
        if delta_hours > 0:
            thi_slope_per_hour = (last["thi"] - first["thi"]) / delta_hours

    expected_points = 6
    valid_points = sum(
        1
        for row in window
        if row.get("temperature_c") is not None and row.get("humidity_pct") is not None
    )
    data_coverage_last_hour = min(1.0, valid_points / expected_points)

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


def build_mold_manager_inputs_from_recent_readings(
    recent_readings: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Derive minimal `mold_manager_step` inputs from recent DB rows.
    """
    if not recent_readings:
        raise ValueError("recent_readings is empty")

    readings_sorted = _sort_readings_newest_first(recent_readings)
    window = readings_sorted[:6]

    latest_valid: Optional[Dict[str, Any]] = None
    for row in window:
        if row.get("temperature_c") is not None and row.get("humidity_pct") is not None:
            latest_valid = row
            break

    if latest_valid is None:
        raise ValueError("No valid row with both temperature_c and humidity_pct in last hour window")

    expected_points = 6
    valid_points = sum(
        1
        for row in window
        if row.get("temperature_c") is not None and row.get("humidity_pct") is not None
    )
    coverage = min(1.0, valid_points / expected_points)

    return {
        "temp_c": float(latest_valid["temperature_c"]),
        "humidity_rh": float(latest_valid["humidity_pct"]),
        "data_coverage_last_hour": coverage,
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


#=============================================================================
# MOLD/CONDENSATION RISK MODEL
#=============================================================================

def dew_point_celsius(temp_c: float, humidity_rh: float) -> float:
    """
    Compute dew point (°C) using the Magnus formula.
    - temp_c: air temperature in °C
    - humidity_rh: relative humidity in % (0..100)

    This is a standard approximation and is good enough for a practical MVP.
    """
    # Guard against invalid humidity
    humidity_rh = max(0.1, min(humidity_rh, 100.0))

    a = 17.62
    b = 243.12  # °C
    gamma = (a * temp_c) / (b + temp_c) + math.log(humidity_rh / 100.0)
    dp = (b * gamma) / (a - gamma)
    return dp


# =============================================================================
# Prevention-first Mold Manager (fan is ON/OFF only)
# =============================================================================
class MoldState(str, Enum):
    SAFE = "SAFE"
    WATCH = "WATCH"
    DANGER = "DANGER"


@dataclass
class MoldParams:
    # Sampling cadence (minutes per window)
    sample_minutes: int = 10

    # Condensation risk is high when air temp is close to dew point:
    # margin = temp - dew_point.
    margin_watch_c: float = 3.0   # <=3°C: early dampness risk (prevention start)
    margin_danger_c: float = 1.5  # <=1.5°C: condensation likely (urgent)

    # RH thresholds (common practical dampness thresholds)
    rh_watch: float = 75.0   # >=70%RH: dampness risk begins
    rh_danger: float = 85.0  # >=85%RH: very mold-friendly

    # Hysteresis thresholds to EXIT states (prevents flapping)
    margin_exit_watch_c: float = 3.5  # need to be clearly safer than watch margin
    rh_exit_watch: float = 70.0


    margin_exit_danger_c: float = 2.0  # improved enough to leave DANGER
    rh_exit_danger: float = 80.0

    # Minimum time spent in a state before allowing state change
    min_minutes_in_state: int = 20  # avoids rapid switching

    # Exposure memory (0..100) — prevention-first:
    # rises fast in danger, rises slower in watch, decays slowly in safe.
    exposure_increase_watch_per_min: float = 0.25   # +15/hour
    exposure_increase_danger_per_min: float = 0.80  # +48/hour
    exposure_decay_safe_per_min: float = 0.10       # -6/hour (slow reset)

    exposure_to_watch: float = 30.0
    exposure_to_danger: float = 70.0

    # Fan automation limits (ON/OFF only)
    min_minutes_between_fan_changes: int = 10  # anti-flapping
    watch_fan_on_minutes: int = 10            # 1 pulse
    watch_fan_off_minutes: int = 20           # rest period
    danger_fan_on_minutes: int = 20           # stronger drying pulse
    danger_fan_off_minutes: int = 10

@dataclass
class MoldMemory:
    """
    Persist per coop (DB). This gives the system memory across time/restarts.
    """
    state: MoldState = MoldState.SAFE
    minutes_in_state: int = 0
    exposure: float = 0.0  # 0..100

    # Fan state tracking (so we can propose stable ON/OFF schedules)
    fan_is_on: bool = False
    minutes_since_fan_change: int = 9999  # start permissive

def mold_manager_step(
    *,
    temp_c: float,
    humidity_rh: float,
    day_night: str,  # "day" or "night"
    memory: MoldMemory,
    params: MoldParams,
) -> Dict[str, Any]:
    """
    One step per sample window. Produces:
      - SAFE/WATCH/DANGER state with hysteresis + minimum time in state
      - exposure (0..100)
      - a simple ON/OFF fan schedule proposal
      - RAG-ready message payload (what to tell the client, exactly)
    """

    # --- Compute dew point / condensation margin
    dp = dew_point_celsius(temp_c, humidity_rh)
    margin_c = temp_c - dp

    # Night increases condensation likelihood slightly (surfaces cool more easily)
    night_margin_bonus = 0.2 if day_night == "night" else 0.0

    # --- Determine zones (vanaf wanneer zit je in de gevarenzone, vanaf wanneer in de waarschuwingszone)
    in_danger_zone = (margin_c <= params.margin_danger_c + night_margin_bonus) or (humidity_rh >= params.rh_danger)
    in_watch_zone = in_danger_zone or (margin_c <= params.margin_watch_c + night_margin_bonus) or (humidity_rh >= params.rh_watch)

    # --- Exposure update (prevention-first)
    # geheugenvariabele die bijhoudt hoe lang het kippenhok al in vochtige omstandigheden zit.
    # stijgt sneller in gevaar, stijgt langzamer in waarschuwingszone, daalt langzaam in veilige zone. ==> dit is hoe vochtopstapeling werkt
    dt = params.sample_minutes
    if in_danger_zone:
        memory.exposure += params.exposure_increase_danger_per_min * dt
    elif in_watch_zone:
        memory.exposure += params.exposure_increase_watch_per_min * dt
    else:
        memory.exposure -= params.exposure_decay_safe_per_min * dt

    memory.exposure = max(0.0, min(memory.exposure, 100.0))

    # --- Candidate target state (climate + exposure memory)
    target = MoldState.SAFE
    if in_danger_zone or memory.exposure >= params.exposure_to_danger:
        target = MoldState.DANGER
    elif in_watch_zone or memory.exposure >= params.exposure_to_watch:
        target = MoldState.WATCH

    # --- Exit conditions (hysteresis)
    can_exit_watch = (margin_c > params.margin_exit_watch_c) and (humidity_rh < params.rh_exit_watch)
    can_exit_danger = (margin_c > params.margin_exit_danger_c) and (humidity_rh < params.rh_exit_danger)

    # --- Minimum time in state
    memory.minutes_in_state += dt
    min_time_met = memory.minutes_in_state >= params.min_minutes_in_state

    prev_state = memory.state
    new_state = memory.state

    if memory.state == MoldState.SAFE:
        if target in {MoldState.WATCH, MoldState.DANGER}:
            new_state = target

    elif memory.state == MoldState.WATCH:
        if target == MoldState.DANGER and min_time_met:
            new_state = MoldState.DANGER
        elif target == MoldState.SAFE and min_time_met and can_exit_watch:
            new_state = MoldState.SAFE

    elif memory.state == MoldState.DANGER:
        if min_time_met and can_exit_danger:
            # downgrade stepwise
            if in_watch_zone or memory.exposure >= params.exposure_to_watch:
                new_state = MoldState.WATCH
            else:
                new_state = MoldState.SAFE

    if new_state != memory.state:
        memory.state = new_state
        memory.minutes_in_state = 0  # reset timer on transition

    # --- Fan schedule proposal (ON/OFF only, anti-flapping)
    memory.minutes_since_fan_change += dt

    desired_fan_on = memory.fan_is_on  # default: keep current to minimize toggles
    fan_reason = "Maintain current fan state to avoid frequent toggling."

    if memory.state == MoldState.SAFE:
        # In safe state we prefer fan OFF unless exposure still elevated.
        # This gives a gentle drying effect without risking over-drying or energy waste.
        if memory.exposure >= params.exposure_to_watch and memory.minutes_since_fan_change >= params.min_minutes_between_fan_changes:
            desired_fan_on = True
            fan_reason = "Exposure still elevated; run a short drying pulse."
        else:
            desired_fan_on = False
            fan_reason = "Conditions safe; keep fan off."

    elif memory.state == MoldState.WATCH:
        # Pulse schedule: ON for watch_fan_on_minutes, then OFF for watch_fan_off_minutes
        cycle = params.watch_fan_on_minutes + params.watch_fan_off_minutes
        phase = (memory.minutes_in_state % cycle)
        desired_fan_on = phase < params.watch_fan_on_minutes
        fan_reason = "WATCH mode: run short ventilation pulses to prevent dampness from persisting."

    elif memory.state == MoldState.DANGER:
        # Stronger pulses
        cycle = params.danger_fan_on_minutes + params.danger_fan_off_minutes
        phase = (memory.minutes_in_state % cycle)
        desired_fan_on = phase < params.danger_fan_on_minutes
        fan_reason = "DANGER mode: stronger ventilation pulses to reduce condensation risk quickly."

    # Apply anti-flapping: only allow change if enough time passed
    fan_command: Optional[Dict[str, Any]] = None
    if desired_fan_on != memory.fan_is_on:
        if memory.minutes_since_fan_change >= params.min_minutes_between_fan_changes:
            memory.fan_is_on = desired_fan_on
            memory.minutes_since_fan_change = 0
            fan_command = {
                "device": "fan",
                "command": "TURN_ON" if desired_fan_on else "TURN_OFF",
                "reason": fan_reason,
            }
        else:
            # Can't change yet; keep current
            desired_fan_on = memory.fan_is_on
            fan_reason += " (Change delayed by anti-flapping rule.)"

    # --- RAG-ready messaging payload (explicit + structured)
    # The RAG should not invent commands; it should repeat these instructions.
    urgency_minutes = 360
    if memory.state == MoldState.DANGER:
        urgency_minutes = 30
    elif memory.state == MoldState.WATCH:
        urgency_minutes = 120

    why_facts = [
        f"Humidity={humidity_rh:.1f}%RH",
        f"DewPoint={dp:.2f}°C",
        f"CondensationMargin={margin_c:.2f}°C (temp - dew point)",
        f"Exposure={memory.exposure:.1f}/100 (slow to reset)",
        f"TimeOfDay={day_night}",
    ]

    # Client effort minimizer: "one-tap" + small checklist.
    # Keep it short and actionable.
    if memory.state == MoldState.SAFE:
        client_actions = [
            "No action needed. System is keeping conditions in the safe band.",
        ]
    elif memory.state == MoldState.WATCH:
        client_actions = [
            "Ventilation will run periodically to keep humidity under control.",
            "Quick check (1 min): ensure bedding is not wet and there are no visible leaks.",
        ]
    else:  # DANGER
        client_actions = [
            "Ventilation will run more frequently to remove moisture.",
            "If bedding is wet, replace/remove the wet part (highest impact).",
            "Check for condensation drip points (roof/corners).",
        ]

    # Automation instruction for the RAG: fan supports only ON/OFF.
    automation_instruction = {
        "fan": "ON" if desired_fan_on else "OFF",
    }

    # Events to emit (log + notify + RAG)
    events: List[Dict[str, Any]] = []

    # Always emit status (for dashboards/debugging)
    events.append({
        "event_type": "MOLD_MANAGER_STATUS",
        "state": memory.state.value,
        "exposure": round(memory.exposure, 1),
        "temp_c": round(temp_c, 1),
        "humidity_rh": round(humidity_rh, 1),
        "dew_point_c": round(dp, 2),
        "condensation_margin_c": round(margin_c, 2),
        "fan_is_on": memory.fan_is_on,
        "why_facts": why_facts,
        "automation_instruction": automation_instruction,
        "client_actions": client_actions,
        "urgency_minutes": urgency_minutes,
    })

    # Emit on state change (notification trigger) ==> only when state changes, to avoid spamming notifications
    if memory.state != prev_state:
        events.append({
            "event_type": "MOLD_ALERT",
            "severity": memory.state.value,
            "message": (
                "SAFE: Conditions are stable."
                if memory.state == MoldState.SAFE else
                "WATCH: Early dampness risk detected. Preventing mold now is easy."
                if memory.state == MoldState.WATCH else
                "DANGER: High condensation/mold risk. Act soon to prevent persistent dampness."
            ),
            "urgency_minutes": urgency_minutes,
            "client_actions": client_actions,
            "automation_instruction": automation_instruction,
            "why_facts": why_facts,
        })

    # Emit fan command request if there is one (strict separation from execution)
    if fan_command is not None:
        events.append({
            "event_type": "AUTOMATION_COMMAND_REQUEST",
            "command": fan_command,
        })

    return {
        "state": memory.state.value,
        "minutes_in_state": memory.minutes_in_state,
        "exposure": round(memory.exposure, 1),
        "dew_point_c": round(dp, 2),
        "condensation_margin_c": round(margin_c, 2),
        "fan_is_on": memory.fan_is_on,
        "fan_command": fan_command,  # None if no change
        "urgency_minutes": urgency_minutes,
        "why_facts": why_facts,  # what RAG should cite
        "client_actions": client_actions,  # what RAG should tell the user
        "events_to_emit": events,
    }


# =============================================================================
# Parameter list (what to tune)
# =============================================================================
"""
Core thresholds (tune carefully):
- margin_watch_c (default 3.0°C):
  Early warning. When air temp is within ~3°C of dew point, surfaces start to stay damp.
- margin_danger_c (default 1.5°C):
  Condensation is likely; dampness can persist and mold conditions become strong.
- rh_watch 80% / rh_danger 90%:
  Practical humidity thresholds for dampness and mold-friendly conditions.

Stability knobs:
- min_minutes_in_state:
  Prevents state ping-pong when values hover near thresholds.
- hysteresis exits (margin_exit_* and rh_exit_*):
  Requires clearly improved conditions before leaving WATCH/DANGER.

Exposure (prevention-first memory):
- exposure increases fast in DANGER, slower in WATCH, decays slowly in SAFE.
- exposure_to_watch / exposure_to_danger:
  Even if RH dips briefly, accumulated dampness keeps the system cautious.

Fan scheduling (ON/OFF only):
- watch_fan_on/off minutes:
  Lightweight drying pulses with low user effort.
- danger_fan_on/off minutes:
  Stronger pulses when condensation risk is high.
- min_minutes_between_fan_changes:
  Avoids rapid toggling of the fan relay.
"""


# =============================================================================
# Integration instructions (short)
# =============================================================================
"""
Persist per coop:
- Store MoldMemory in DB: coop_id, state, minutes_in_state, exposure, fan_is_on, minutes_since_fan_change, updated_at

Log:
- Log MOLD_MANAGER_STATUS every sample window (for debugging and charts)
- Log MOLD_ALERT only on state transitions (for notifications)

Automation execution:
- Do NOT let the LLM directly flip relays.
- Your backend should listen for AUTOMATION_COMMAND_REQUEST and apply policy checks:
    - manual override?
    - relay cooldown?
    - device health?
  Then execute TURN_ON / TURN_OFF.

RAG instructions (important):
- Feed the RAG the full MOLD_MANAGER_STATUS payload.
- In the prompt, require:
    1) A 1-sentence summary of state + urgency_minutes
    2) A "Why" list that ONLY uses why_facts (no hallucination)
    3) The client_actions list verbatim (minimal effort)
    4) If automation_instruction.fan is ON, state exactly: "Fan should be ON."
       If automation_instruction.fan is OFF, state exactly: "Fan should be OFF."
"""
