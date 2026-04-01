"""
ventilation_system.py  —  Smart Chicken Coop (ChickenCoopComfort)
==================================================================
Called by the Raspberry Pi when the risk-assessment system decides
ventilation is needed. Does three things:

  1. Read sensor data from Supabase
  2. Read outdoor weather from Open-Meteo
  3. Compute and command the fan speed

The on/off decision is made ELSEWHERE. This script only answers:
HOW HARD should the fan blow?

Priority order
--------------
  BASE  CO2 ratio-inversion  — always the floor rate (gas safety)
  1.    Heat stress           — increase if outside is cool enough to help
  2.    Humidity              — increase if outside is dry enough to help
  3.    Cold stress           — clamp back down to CO2 floor
  HARD  H2S emergency        — immediate full-speed override, ignores everything

Outdoor capacity checks
-----------------------
Before applying heat or humidity corrections we verify that ventilating
actually helps:
  - Cooling only activates when T_amb < T_in  (outside is actually cooler)
  - Drying  only activates when AH_out < AH_in (outside air is actually drier)
If the outdoor air cannot help, the correction is skipped and a note is logged.

CO2 sensor
----------
Column co2_ppm is read if present. When None (sensor not yet wired) the
script falls back to a model-based seed rate from bird count and weight.
Once the sensor is live, ratio-inversion takes over automatically.

Environment variables (.env file next to this script)
------------------------------------------------------
  SUPABASE_URL    https://qdwofrcncjnhstbqegnj.supabase.co
  SUPABASE_KEY    sb_publishable_5nKBAUpvEiD-E6diKaf6dA_jBSEb47Z
  COOP_LATITUDE   50.864403
  COOP_LONGITUDE  4.686699

Install:  pip install supabase requests python-dotenv
"""

import json
import logging
import math
import os
from datetime import datetime, timezone
from pathlib import Path

import requests
from supabase import create_client, Client

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
log = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — SETTINGS
# ═══════════════════════════════════════════════════════════════════════════════

COOP_LAT    = float(os.environ.get("COOP_LATITUDE",  "50.864403"))
COOP_LON    = float(os.environ.get("COOP_LONGITUDE", "4.686699"))

# Hardware
VENT_MAX    = 150.0   # m3/h — confirmed fan maximum
VENT_MIN    = 0.0     # m3/h — no enforced minimum yet
MAX_SLEW    = 50.0    # m3/h — max change per cycle (prevents sudden jumps)

# Setpoints
T_MIN       = 16.0    # °C  — cold stress threshold
T_MAX       = 22.0    # °C  — heat stress threshold
RH_MAX      = 0.70    # 0-1 — humidity limit
CO2_TARGET  = 2000.0  # ppm — desired indoor CO2
CO2_AMBIENT = 400.0   # ppm — outdoor baseline CO2

# Heat-risk-based boost (small supervisory correction)
HEAT_RISK_BOOST_1 = 50.0   # mild boost
HEAT_RISK_BOOST_2 = 75.0   # medium boost
HEAT_RISK_BOOST_3 = 90.0   # high boost

HEAT_RISK_MULT_LOW  = 1.05   # +5%
HEAT_RISK_MULT_MID  = 1.10   # +10%
HEAT_RISK_MULT_HIGH = 1.15   # +15%

# H2S
H2S_WARN    = 1.0     # ppm — warning: boost ventilation proportionally
H2S_EMERG   = 5.0     # ppm — hard override: fan to maximum regardless of all else

# Flock defaults (used for model-based seed when CO2 sensor absent)
BIRD_WEIGHT_KG  = 2.5
CO2_PER_BIRD_LD = 3.8   # L CO2 / day / bird (at reference conditions)

STATE_FILE  = Path("vent_state.json")


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — SUPABASE  (read only)
# ═══════════════════════════════════════════════════════════════════════════════

def connect() -> Client:
    url = os.environ.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_KEY")
    if not url or not key:
        raise EnvironmentError(
            "SUPABASE_URL and SUPABASE_KEY must be set in your .env file."
        )
    return create_client(url, key)


def check_connection(client: Client) -> bool:
    """Ping each required table and log ✓ / ✗. Returns False if any fail."""
    log.info("── Supabase connection check ────────────────────────────")
    ok = True
    for table in ["sensor_readings_colson", "cv_counts_colson"]:
        try:
            client.table(table).select("id").limit(1).execute()
            log.info("  ✓  %s", table)
        except Exception as e:
            log.error("  ✗  %s  →  %s", table, e)
            ok = False
    log.info("── %s ─────────────────────────────────────────────────",
             "OK" if ok else "FAILED")
    return ok


def read_sensors(client: Client) -> dict:
    """
    Latest row from sensor_readings_colson.

    Columns read
    ------------
    temperature_c   → T_in  [°C]
    humidity_pct    → RH_in [0-1]  (stored as %, converted here)
    h2s_ppm         → H2S_in [ppm]
    co2_ppm         → CO2_in [ppm] or None if column absent / null

    All other columns (h2s_level, mold_risk_score, ventilation_on, ...) are
    ignored here — they are used by other scripts.
    """
    resp = (
        client.table("sensor_readings_colson")
        .select("*")
        .order("timestamp", desc=True)
        .limit(1)
        .execute()
    )
    if not resp.data:
        raise ValueError("sensor_readings_colson is empty.")
    row = resp.data[0]

    co2_raw = row.get("co2_ppm")
    sensors = {
        "T_in":   float(row["temperature_c"]),
        "RH_in":  float(row["humidity_pct"]) / 100.0,
        "H2S_in": float(row.get("h2s_ppm") or 0.0),
        "CO2_in": float(co2_raw) if co2_raw is not None else None,
    }
    log.info(
        "Sensors   T=%.1f°C  RH=%.0f%%  H2S=%.2f ppm  CO2=%s",
        sensors["T_in"], sensors["RH_in"] * 100, sensors["H2S_in"],
        f"{sensors['CO2_in']:.0f} ppm" if sensors["CO2_in"] is not None
        else "no sensor yet",
    )
    return sensors


def read_bird_count(client: Client) -> int:
    """Latest chicken count from cv_counts_colson. Returns 0 if empty."""
    resp = (
        client.table("cv_counts_colson")
        .select("number_of_chickens, egg_count, timestamp")
        .order("timestamp", desc=True)
        .limit(1)
        .execute()
    )
    if not resp.data:
        log.warning("cv_counts_colson empty — defaulting to 0 birds")
        return 0
    row = resp.data[0]
    n = int(row.get("number_of_chickens") or 0)
    log.info("CV count  n_birds=%d  eggs=%d", n, int(row.get("egg_count") or 0))
    return n


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — WEATHER  (Open-Meteo, free, no API key)
# ═══════════════════════════════════════════════════════════════════════════════

def read_weather() -> tuple[float, float]:
    """
    Current outdoor T [°C] and RH [0-1] from Open-Meteo.
    Falls back to (10.0, 0.65) on failure so the script keeps running.
    """
    try:
        r = requests.get(
            "https://api.open-meteo.com/v1/forecast",
            params={
                "latitude":      COOP_LAT,
                "longitude":     COOP_LON,
                "current":       "temperature_2m,relative_humidity_2m",
                "forecast_days": 1,
            },
            timeout=10,
        )
        r.raise_for_status()
        c = r.json()["current"]
        T_amb  = float(c["temperature_2m"])
        RH_amb = float(c["relative_humidity_2m"]) / 100.0
        log.info("Weather   T_amb=%.1f°C  RH_amb=%.0f%%", T_amb, RH_amb * 100)
        return T_amb, RH_amb
    except Exception as e:
        log.warning("Weather fetch failed (%s) — fallback T=10°C RH=65%%", e)
        return 10.0, 0.65


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — PHYSICS HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def absolute_humidity(T: float, RH: float) -> float:
    """Absolute humidity [kg water / kg dry air] — Buck equation."""
    Psat = 0.61121 * math.exp((18.678 - T / 234.5) * (T / (257.14 + T)))
    return 0.622 * (Psat * RH / (101.325 - Psat * RH))


def air_density(T: float) -> float:
    """Dry air density [kg/m3]."""
    return 353.0 / (T + 273.15)


def latent_heat(T: float) -> float:
    """Latent heat of vaporisation [J/g] — lookup table interpolation."""
    pts = [
        (0, 2500.9), (2, 2496.4), (4, 2491.2), (10, 2477.2), (14, 2467.7),
        (18, 2458.3), (20, 2453.5), (25, 2441.7), (30, 2429.8), (34, 2420.3),
        (40, 2406.4), (44, 2396.4), (50, 2381.9), (54, 2372.3), (60, 2357.7),
        (70, 2333.7), (80, 2308.0), (90, 2282.5), (96, 2266.9),
    ]
    if T <= pts[0][0]:  return pts[0][1]
    if T >= pts[-1][0]: return pts[-1][1]
    for i in range(len(pts) - 1):
        t0, l0 = pts[i]; t1, l1 = pts[i + 1]
        if t0 <= T <= t1:
            return l0 + (l1 - l0) * (T - t0) / (t1 - t0)


def bird_heat_production(n: int, W: float, T: float) -> tuple[float, float]:
    """
    Total sensible heat [W] and moisture [g/s] for n birds of weight W [kg]
    at indoor temperature T [°C].
    """
    total    = 10.62 * (W ** 0.75)
    sensible = (0.61 * (1000 + 20 * (20 - T) - 0.228 * T**2)) * (total / 1000)
    latent_w = total - sensible
    moisture = latent_w / latent_heat(T)   # g/s per bird
    return sensible * n, moisture * n


def co2_seed_rate(n_birds: int) -> float:
    """
    Model-based CO2 ventilation rate [m3/h] — used only when no CO2
    sensor is available. Based on bird count and average CO2 production.
    Once co2_ppm is in Supabase, ratio-inversion replaces this entirely.
    """
    q_m3h = (n_birds * CO2_PER_BIRD_LD) / (24 * 1000)   # m3/s equivalent
    delta  = (CO2_TARGET - CO2_AMBIENT) * 1e-6
    if delta <= 0:
        return VENT_MAX
    rate = q_m3h / delta
    log.info("CO2 seed  model-based (no sensor) → %.0f m3/h", rate)
    return rate


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — OUTDOOR CAPACITY CHECKS
# ═══════════════════════════════════════════════════════════════════════════════

def can_cool(T_in: float, T_amb: float) -> bool:
    """
    Ventilation can only cool the coop when outside is cooler than inside.
    If T_amb >= T_in, pumping in outside air makes things worse.
    """
    if T_amb < T_in:
        return True
    log.info("Cooling   SKIP — T_amb=%.1f°C >= T_in=%.1f°C (outside warmer)", T_amb, T_in)
    return False


def can_dry(T_in: float, RH_in: float, T_amb: float, RH_amb: float) -> bool:
    """
    Ventilation can only reduce indoor humidity when outdoor absolute
    humidity is lower than indoor absolute humidity.
    If outdoor air is equally or more humid, bringing it in adds moisture.
    """
    AH_in  = absolute_humidity(T_in,  RH_in)
    AH_out = absolute_humidity(T_amb, RH_amb)
    if AH_out < AH_in:
        return True
    log.info(
        "Drying    SKIP — outdoor AH=%.4f >= indoor AH=%.4f (outside too humid)",
        AH_out, AH_in,
    )
    return False


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 6 — VENTILATION CONTROL
# ═══════════════════════════════════════════════════════════════════════════════

def compute_fan_rate(
    sensors:     dict,
    heat_risk:   dict,
    T_amb:       float,
    RH_amb:      float,
    n_birds:     int,
    prev_rate:   float,
    initialised: bool,
) -> tuple[float, str]:
    """
    Compute fan rate [m3/h] for this cycle.

    Returns (rate, reason_string).

    Full priority order
    -------------------
    HARD  H2S >= H2S_EMERG   → VENT_MAX, no questions asked
    BASE  CO2 ratio-inversion → always the minimum floor
    1.    Heat stress         → raise above floor if outdoor can cool
    2.    Humidity            → raise above floor if outdoor can dry
    3.    Cold stress         → clamp everything back to CO2 floor
    """
    T_in   = sensors["T_in"]
    RH_in  = sensors["RH_in"]
    H2S_in = sensors["H2S_in"]
    CO2_in = sensors["CO2_in"]   # None until sensor is installed
    heat_risk_score = float(heat_risk["risk_score"])

    notes = []

    # Seed on very first run
    if not initialised:
        prev_rate = VENT_MAX / 4.0
        notes.append(f"cold-start seed {prev_rate:.0f} m3/h")

    # ── HARD OVERRIDE: H2S emergency ─────────────────────────────────
    # Checked before everything else. No capacity check, no slew limit.
    if H2S_in >= H2S_EMERG:
        return VENT_MAX, f"H2S EMERGENCY {H2S_in:.1f} ppm — fan at maximum"

    # ── BASE: CO2 floor ───────────────────────────────────────────────
    # This is always the minimum acceptable rate. All other corrections
    # can only increase it, never go below it (except cold stress).
    if CO2_in is not None:
        # Ratio-inversion: if CO2 is 10% above target we need 10% more air
        num = CO2_in    - CO2_AMBIENT
        den = CO2_TARGET - CO2_AMBIENT
        if den > 0 and num > 0:
            vr_co2 = prev_rate * (num / den)
        else:
            vr_co2 = VENT_MIN   # CO2 at or below target — no CO2 ventilation needed
        notes.append(f"CO2={CO2_in:.0f} ppm → floor {vr_co2:.0f} m3/h")
    else:
        # No sensor yet: use the bird-count model as the floor
        vr_co2 = co2_seed_rate(n_birds)
        notes.append(f"CO2 model seed (no sensor) → floor {vr_co2:.0f} m3/h")

    # Clamp CO2 floor to hardware limits
    vr_co2 = max(VENT_MIN, min(vr_co2, VENT_MAX))

    # Start target at the CO2 floor
    target = vr_co2

    # ── PRIORITY 1: Heat stress ───────────────────────────────────────
    if T_in > T_MAX:
        if can_cool(T_in, T_amb):
            # Heat-balance inversion: how much airflow removes sensible bird heat?
            Q_sen, _ = bird_heat_production(n_birds, BIRD_WEIGHT_KG, T_in)
            rho       = air_density(T_amb)
            cp        = 1005.0        # J / (kg·K)
            dT        = T_MAX - T_amb
            if dT > 0:
                vr_temp = (Q_sen / (rho * cp * dT)) * 3600
                # Small heat-risk-based boost on top of the physical heat target
                if heat_risk_score >= HEAT_RISK_BOOST_3:
                    heat_boost = HEAT_RISK_MULT_HIGH
                elif heat_risk_score >= HEAT_RISK_BOOST_2:
                    heat_boost = HEAT_RISK_MULT_MID
                elif heat_risk_score >= HEAT_RISK_BOOST_1:
                    heat_boost = HEAT_RISK_MULT_LOW
                else:
                    heat_boost = 1.0

                vr_temp_boosted = vr_temp * heat_boost

                if vr_temp_boosted > target:
                    notes.append(
                        f"heat stress T={T_in:.1f}°C risk={heat_risk_score:.1f} "
                        f"→ {vr_temp_boosted:.0f} m3/h "
                        f"(base {vr_temp:.0f}, boost x{heat_boost:.2f}, "
                        f"T_amb={T_amb:.1f}°C, can cool)"
                    )
                    target = vr_temp_boosted
        # else: can_cool() already logged the skip reason

    # ── PRIORITY 2: Humidity ──────────────────────────────────────────
    if RH_in > RH_MAX:
        if can_dry(T_in, RH_in, T_amb, RH_amb):
            AH_in  = absolute_humidity(T_in,  RH_in)
            AH_out = absolute_humidity(T_amb, RH_amb)
            AH_tgt = absolute_humidity(T_in,  RH_MAX)
            num = AH_in  - AH_out
            den = AH_tgt - AH_out
            if den > 0 and num > 0:
                vr_rh = prev_rate * (num / den)
                if vr_rh > target:
                    notes.append(
                        f"humidity RH={RH_in*100:.0f}% → {vr_rh:.0f} m3/h "
                        f"(outdoor AH lower, can dry)"
                    )
                    target = vr_rh
        # else: can_dry() already logged the skip reason

    # ── H2S warning boost (on top of whatever target we reached) ─────
    if H2S_WARN <= H2S_in < H2S_EMERG:
        boost  = VENT_MAX * 0.15 * (H2S_in - H2S_WARN) / max(H2S_WARN, 1e-9)
        target = min(target + boost, VENT_MAX)
        notes.append(f"H2S warning {H2S_in:.1f} ppm → boost +{boost:.0f} m3/h")

    # ── PRIORITY 3: Cold stress — clamp back to CO2 floor ────────────
    # Cold stress never goes below the CO2 floor (gas safety).
    if T_in < T_MIN:
        if target > vr_co2:
            target = vr_co2
            notes.append(
                f"cold stress T={T_in:.1f}°C — clamped to CO2 floor {vr_co2:.0f} m3/h"
            )

    # ── Slew rate limit ───────────────────────────────────────────────
    delta = max(-MAX_SLEW, min(target - prev_rate, MAX_SLEW))
    rate  = max(VENT_MIN, min(prev_rate + delta, VENT_MAX))

    return rate, " | ".join(notes) if notes else "baseline"


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 7 — STATE  (persisted between 30-min cron runs)
# ═══════════════════════════════════════════════════════════════════════════════

def load_state() -> tuple[float, bool]:
    if STATE_FILE.exists():
        try:
            s = json.loads(STATE_FILE.read_text())
            return float(s["prev_rate"]), bool(s["initialised"])
        except Exception:
            pass
    return 0.0, False


def save_state(rate: float) -> None:
    STATE_FILE.write_text(
        json.dumps({"prev_rate": rate, "initialised": True}, indent=2)
    )


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 8 — FAN ACTUATOR  (replace set_rate() with your hardware call)
# ═══════════════════════════════════════════════════════════════════════════════

class Fan:
    """
    Stub — replace set_rate() with your real hardware interface.

    GPIO PWM (RPi):  fan_pwm.ChangeDutyCycle(rate / VENT_MAX * 100)
    MQTT:            mqtt.publish("coop/fan/speed", rate)
    HTTP relay:      requests.post("http://relay/fan", json={"rate": rate})
    """
    def set_rate(self, rate: float) -> None:
        log.info("FAN COMMAND → %.0f m3/h", rate)   # ← replace with real call


# ═══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    log.info("════ Ventilation cycle %s ════",
             datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"))

    # Step 1 — Supabase
    client = connect()
    if not check_connection(client):
        log.error("Aborting — fix Supabase connection.")
        return

    try:
        sensors = read_sensors(client)
    except ValueError as e:
        log.error("Sensor read failed: %s", e)
        return

    n_birds = read_bird_count(client)

    # Step 2 — Weather
    T_amb, RH_amb = read_weather()

    # Step 3 — Control
    prev_rate, initialised = load_state()
    rate, reason = compute_fan_rate(
        sensors, {"risk_score": 0.0}, T_amb, RH_amb, n_birds, prev_rate, initialised
    )

    log.info("Result    rate=%.0f m3/h", rate)
    log.info("Reason    %s", reason)

    Fan().set_rate(rate)
    save_state(rate)

    log.info("════ Cycle complete ════")


if __name__ == "__main__":
    main()
