"""
ventilation_system.py  —  Smart Chicken Coop (ChickenCoopComfort)
==================================================================
Single-file ventilation system. Run every 30 minutes via cron:

    */30 * * * *  cd /opt/coop && python ventilation_system.py >> logs/vent.log 2>&1

What happens each run
---------------------
  1. Check Supabase connection — abort with a clear message if it fails.
  2. Read latest row from sensor_readings_colson.
  3. Read latest bird count from cv_counts_colson.
  4. Read setpoints from flock_config.
  5. Score environmental risk (temp, humidity, H2S) → write to risk_assessments.
     If risk is below threshold AND no H2S emergency → exit, fan stays off.
  6. Fetch outdoor weather from Open-Meteo (cached 30 min in weather_cache).
  7. Run ventilation control logic (ratio-inversion, no PI loop).
  8. Command the fan actuator.
  9. Write result to ventilation_log.
  10. Persist controller state to vent_state.json for the next 30-min cycle.

Real Supabase tables used
--------------------------
  sensor_readings_colson   Live sensor data from Pi / ESP32
    timestamp              freshness check
    temperature_c          → T_in  [°C]
    humidity_pct           → RH_in [%] converted to 0-1
    h2s_ppm                → H2S_in [ppm]
    h2s_level              text label, logged only
    mold_risk_score        float, logged only
    ventilation_on         bool, logged only

  cv_counts_colson         Bird count from computer-vision camera
    timestamp
    number_of_chickens     → n_birds used in heat/moisture calculations
    egg_count              logged only

  flock_config             Setpoints — edit in Supabase dashboard
    active, bird_weight_kg,
    T_min, T_max, RH_max,
    H2S_warning, H2S_emergency,
    vent_max_m3h, vent_min_m3h, max_slew_m3h

  risk_assessments         Written by this script each cycle
  ventilation_log          Written by this script on activation
  weather_cache            One row, upserted by this script each cycle

No CO2 sensor
-------------
  CO2 control is bypassed. When a CO2 sensor is added, map its column
  to CO2_in and re-enable CO2 scoring in calculate_risk().

Hardware limits
---------------
  vent_max : 150 m3/h  (set in flock_config, fallback hardcoded below)
  vent_min : 0 m3/h    (no enforced minimum until spec is confirmed)

Environment variables  (put in a .env file next to this script)
--------------------------------------------------------------
  SUPABASE_URL            https://xxxx.supabase.co
  SUPABASE_KEY            your-service-role-key
  COOP_LATITUDE           50.88   (update to your actual coop GPS location)
  COOP_LONGITUDE          4.70
  RISK_TRIGGER_THRESHOLD  0.45
  VENT_STATE_FILE         vent_state.json

Install
-------
  pip install supabase requests python-dotenv

SQL for new tables
------------------
  See APPENDIX at the bottom of this file.
"""

import json
import logging
import math
import os
from datetime import datetime, timezone
from pathlib import Path

import requests
from supabase import create_client, Client

# Load .env file if present (silently ignored if python-dotenv not installed)
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
# SECTION 1 — CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

DEFAULT_LATITUDE       = float(os.environ.get("COOP_LATITUDE",  "50.864403"))
DEFAULT_LONGITUDE      = float(os.environ.get("COOP_LONGITUDE", "4.686699"))
CACHE_MAX_AGE_MINUTES  = 30
RISK_TRIGGER_THRESHOLD = float(os.environ.get("RISK_TRIGGER_THRESHOLD", "0.45"))
STATE_FILE             = Path(os.environ.get("VENT_STATE_FILE", "vent_state.json"))
OPEN_METEO_URL         = "https://api.open-meteo.com/v1/forecast"

# Hardware fallbacks — override via flock_config table in Supabase
VENT_MAX_FALLBACK      = 150.0   # m3/h — confirmed fan maximum
VENT_MIN_FALLBACK      = 0.0    # m3/h — no minimum enforced yet

# ── Column names for sensor_readings_colson ──────────────────────────────────
# Update these constants if your Supabase column names ever change
SENSOR_TABLE          = "sensor_readings_colson"
SENSOR_COL_TIMESTAMP  = "timestamp"
SENSOR_COL_TEMP       = "temperature_c"
SENSOR_COL_HUMIDITY   = "humidity_pct"       # stored 0-100, converted to 0-1
SENSOR_COL_H2S_PPM    = "h2s_ppm"
SENSOR_COL_H2S_LEVEL  = "h2s_level"          # text label, logged only
SENSOR_COL_MOLD_SCORE = "mold_risk_score"    # logged only
SENSOR_COL_VENT_ON    = "ventilation_on"     # logged only

# ── Column names for cv_counts_colson ────────────────────────────────────────
BIRD_TABLE            = "cv_counts_colson"
BIRD_COL_TIMESTAMP    = "timestamp"
BIRD_COL_COUNT        = "number_of_chickens"
BIRD_COL_EGGS         = "egg_count"           # logged only

SENSOR_RANGES = {
    "T_in":   (-15.0, 50.0),
    "RH_in":  (0.001,  1.0),    # after /100 conversion
    "H2S_in": (  0.0, 100.0),
    "T_amb":  (-20.0, 45.0),
    "RH_amb": (0.001,  1.0),
}


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — SUPABASE CONNECTION  (with explicit connection check)
# ═══════════════════════════════════════════════════════════════════════════════

def get_db_client() -> Client:
    """
    Create a Supabase client from environment variables.
    Raises EnvironmentError immediately if credentials are missing.
    """
    url = os.environ.get('https://qdwofrcncjnhstbqegnj.supabase.co')
    key = os.environ.get('sb_publishable_5nKBAUpvEiD-E6diKaf6dA_jBSEb47Z')
    if not url or not key:
        raise EnvironmentError(
            "\n"
            "  SUPABASE_URL and SUPABASE_KEY are not set.\n"
            "  Add them to a .env file next to this script:\n"
            "\n"
            "      SUPABASE_URL=https://xxxx.supabase.co\n"
            "      SUPABASE_KEY=your-service-role-key\n"
        )
    return create_client(url, key)


def check_connection(client: Client) -> bool:
    """
    Ping Supabase and verify every required table is reachable.
    Logs a clear ✓ / ✗ status line for each table.
    Returns True only when all tables respond — the main cycle
    aborts immediately if this returns False.
    """
    log.info("─── Supabase connection check ───────────────────────────────")
    required_tables = [
        SENSOR_TABLE,
        BIRD_TABLE,
        "flock_config",
    ]
    all_ok = True
    for table in required_tables:
        try:
            client.table(table).select("id").limit(1).execute()
            log.info("  ✓  %-30s reachable", table)
        except Exception as exc:
            log.error("  ✗  %-30s FAILED: %s", table, exc)
            all_ok = False

    if all_ok:
        log.info("─── All tables reachable — connection OK ────────────────────")
    else:
        log.error("─── Connection check FAILED ─────────────────────────────────")
        log.error("    Check SUPABASE_URL, SUPABASE_KEY, and that every table")
        log.error("    listed above exists in schema 'public'.")
    return all_ok


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — SUPABASE READ / WRITE HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def fetch_latest_sensor_reading(client: Client) -> dict:
    """
    Read the most recent row from sensor_readings_colson and normalise
    column names to the internal convention used throughout this script.

    Returned keys
    -------------
    T_in      float  °C
    RH_in     float  0-1   (converted from humidity_pct stored as 0-100)
    H2S_in    float  ppm
    timestamp str
    h2s_level str    text label (logged only)
    mold_risk float  mold_risk_score (logged only)
    vent_on   bool   ventilation_on flag from Supabase (logged only)
    """
    resp = (
        client.table(SENSOR_TABLE)
        .select("*")
        .order(SENSOR_COL_TIMESTAMP, desc=True)
        .limit(1)
        .execute()
    )
    if not resp.data:
        raise ValueError(
            f"Table '{SENSOR_TABLE}' returned no rows. "
            "Check that your Pi / ESP32 is writing sensor data."
        )
    row = resp.data[0]
    return {
        "T_in":      float(row[SENSOR_COL_TEMP]),
        "RH_in":     float(row[SENSOR_COL_HUMIDITY]) / 100.0,
        "H2S_in":    float(row.get(SENSOR_COL_H2S_PPM) or 0.0),
        "timestamp": str(row.get(SENSOR_COL_TIMESTAMP, "")),
        "h2s_level": str(row.get(SENSOR_COL_H2S_LEVEL, "")),
        "mold_risk": float(row.get(SENSOR_COL_MOLD_SCORE) or 0.0),
        "vent_on":   bool(row.get(SENSOR_COL_VENT_ON, False)),
    }


def fetch_latest_bird_count(client: Client) -> dict:
    """
    Read the most recent row from cv_counts_colson.

    Returned keys
    -------------
    n_birds   int   number_of_chickens  (used in heat/moisture calculations)
    egg_count int   (logged only)
    timestamp str
    """
    resp = (
        client.table(BIRD_TABLE)
        .select("*")
        .order(BIRD_COL_TIMESTAMP, desc=True)
        .limit(1)
        .execute()
    )
    if not resp.data:
        log.warning(
            "'%s' is empty — defaulting to 0 birds. "
            "Heat/moisture calculations will be skipped until the camera feeds data.",
            BIRD_TABLE,
        )
        return {"n_birds": 0, "egg_count": 0, "timestamp": ""}

    row = resp.data[0]
    return {
        "n_birds":   int(row.get(BIRD_COL_COUNT) or 0),
        "egg_count": int(row.get(BIRD_COL_EGGS)  or 0),
        "timestamp": str(row.get(BIRD_COL_TIMESTAMP, "")),
    }


def fetch_flock_config(client: Client) -> dict:
    """
    Read the active row from flock_config.
    Required columns: active, bird_weight_kg, T_min, T_max, RH_max,
                      H2S_warning, H2S_emergency
    Optional columns: vent_max_m3h, vent_min_m3h, max_slew_m3h
    """
    resp = (
        client.table("flock_config")
        .select("*")
        .eq("active", True)
        .limit(1)
        .execute()
    )
    if not resp.data:
        raise ValueError(
            "No active row found in flock_config. "
            "Add a row with active=true in the Supabase dashboard."
        )
    return resp.data[0]


def fetch_cached_weather(client: Client) -> dict | None:
    """Return the single cached weather row, or None if not yet populated."""
    resp = (
        client.table("weather_cache")
        .select("*")
        .order("fetched_at", desc=True)
        .limit(1)
        .execute()
    )
    return resp.data[0] if resp.data else None


def upsert_weather_cache(client: Client, T_amb: float, RH_amb: float,
                         description: str) -> None:
    """Writes disabled — logs only. Re-enable when ready to persist weather."""
    log.info("WRITE SUPPRESSED — weather_cache: %s", description)


def insert_risk_assessment(client: Client, assessment: dict) -> None:
    """Writes disabled — logs only. Re-enable when ready to persist risk scores."""
    log.info("WRITE SUPPRESSED — risk_assessments: level=%s score=%.3f",
             assessment.get("risk_level"), assessment.get("overall_score"))


def insert_ventilation_log(client: Client, rate: float, diag: dict,
                           risk_level: str, n_birds: int,
                           T_amb: float, RH_amb: float) -> None:
    """Writes disabled — logs only. Re-enable when ready to persist vent log."""
    log.info("WRITE SUPPRESSED — ventilation_log: rate=%.0f m3/h  limiting=%s  risk=%s",
             rate, diag.get("limiting"), risk_level)


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — WEATHER  (Open-Meteo, free — no API key required)
# ═══════════════════════════════════════════════════════════════════════════════

_WMO_CODES = {
    0: "Clear sky", 1: "Mainly clear", 2: "Partly cloudy", 3: "Overcast",
    45: "Fog", 48: "Icy fog",
    51: "Light drizzle", 53: "Moderate drizzle", 55: "Dense drizzle",
    61: "Slight rain",   63: "Moderate rain",    65: "Heavy rain",
    71: "Slight snow",   73: "Moderate snow",    75: "Heavy snow",
    80: "Slight showers", 81: "Moderate showers", 82: "Violent showers",
    95: "Thunderstorm",  96: "Thunderstorm + slight hail",
    99: "Thunderstorm + heavy hail",
}


def fetch_weather(lat: float = DEFAULT_LATITUDE,
                  lon: float = DEFAULT_LONGITUDE,
                  timeout: int = 10) -> tuple[float, float, str]:
    """
    Call Open-Meteo current-conditions endpoint.
    Returns (T_amb [°C], RH_amb [0-1], human-readable description).
    No API key needed — completely free.
    """
    resp = requests.get(OPEN_METEO_URL, params={
        "latitude":      lat,
        "longitude":     lon,
        "current":       "temperature_2m,relative_humidity_2m,weather_code",
        "forecast_days": 1,
    }, timeout=timeout)
    resp.raise_for_status()
    current     = resp.json().get("current", {})
    T_amb       = float(current["temperature_2m"])
    RH_amb      = float(current["relative_humidity_2m"]) / 100.0
    code        = current.get("weather_code", -1)
    description = (
        f"{_WMO_CODES.get(code, f'Code {code}')} "
        f"| T={T_amb:.1f}°C RH={RH_amb*100:.0f}%"
    )
    log.info("Weather fetched from Open-Meteo: %s", description)
    return T_amb, RH_amb, description


def get_ambient_conditions(client: Client,
                            lat: float = DEFAULT_LATITUDE,
                            lon: float = DEFAULT_LONGITUDE) -> tuple[float, float]:
    """
    Return (T_amb [°C], RH_amb [0-1]).

    Uses the Supabase weather_cache if the cached value is younger than
    CACHE_MAX_AGE_MINUTES (30 min). Otherwise fetches fresh data from
    Open-Meteo and updates the cache. Falls back to stale cache on API error.
    """
    cached = fetch_cached_weather(client)
    if cached:
        fetched_at = datetime.fromisoformat(cached["fetched_at"])
        if fetched_at.tzinfo is None:
            fetched_at = fetched_at.replace(tzinfo=timezone.utc)
        age_min = (datetime.now(timezone.utc) - fetched_at).total_seconds() / 60
        if age_min < CACHE_MAX_AGE_MINUTES:
            log.info("Weather: using Supabase cache (%.0f min old) — "
                     "T=%.1f°C RH=%.0f%%", age_min,
                     cached["T_amb"], cached["RH_amb"] * 100)
            return float(cached["T_amb"]), float(cached["RH_amb"])

    try:
        T_amb, RH_amb, description = fetch_weather(lat=lat, lon=lon)
        upsert_weather_cache(client, T_amb, RH_amb, description)
        return T_amb, RH_amb
    except Exception as exc:
        log.warning("Weather API failed (%s) — falling back to stale cache", exc)
        if cached:
            return float(cached["T_amb"]), float(cached["RH_amb"])
        raise RuntimeError(
            "Weather unavailable and no cache exists. "
            "Check internet connection and COOP_LATITUDE / COOP_LONGITUDE."
        ) from exc


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — RISK CALCULATOR  (gates ventilation on/off)
# ═══════════════════════════════════════════════════════════════════════════════
#dit stuk wordt ultiem dan nog aangepast om de trigger naar het risk calculation script te zetten
def _ramp(value: float, safe: float, warn: float, critical: float,
          direction: str = "high") -> float:
    """
    Piecewise linear risk score 0-1.
      0.0  = within safe bounds
      0.5  = at warning level
      1.0  = at or beyond critical level
    direction='low' flips the scale (used for cold-stress scoring).
    """
    if direction == "low":
        value, safe, warn, critical = -value, -safe, -warn, -critical
    if value <= safe:     return 0.0
    if value >= critical: return 1.0
    if value <= warn:
        return 0.5 * (value - safe) / (warn - safe)
    return 0.5 + 0.5 * (value - warn) / (critical - warn)


def calculate_risk(sensor: dict, config: dict) -> dict:
    """
    Score temperature, humidity and H2S risk dimensions (each 0-1).
    CO2 scoring omitted — no sensor installed yet.

    overall_score = max across all dimensions (worst-case logic).
    ventilation_needed = True when overall_score >= RISK_TRIGGER_THRESHOLD.

    Returns a dict ready to insert into risk_assessments.
    """
    r_rh  = _ramp(sensor["RH_in"],
                  config["RH_max"],
                  config["RH_max"] + 0.07,
                  config["RH_max"] + 0.15,  "high")

    r_th  = _ramp(sensor["T_in"],
                  config["T_max"],
                  config["T_max"] + 1.5,
                  config["T_max"] + 4.0,    "high")

    r_tl  = _ramp(sensor["T_in"],
                  config["T_min"],
                  config["T_min"] - 2.0,
                  config["T_min"] - 5.0,    "low")

    r_h2s = _ramp(sensor["H2S_in"],
                  0.0,
                  config["H2S_warning"],
                  config["H2S_emergency"],   "high")

    temp_risk = max(r_th, r_tl)
    overall   = max(r_rh, temp_risk, r_h2s)

    if overall >= 0.85:   level = "critical"
    elif overall >= 0.60: level = "high"
    elif overall >= 0.35: level = "medium"
    else:                 level = "low"

    notes = []
    if r_rh  > 0.4: notes.append(f"RH={sensor['RH_in']*100:.0f}% (risk={r_rh:.2f})")
    if r_th  > 0.4: notes.append(f"T_high={sensor['T_in']:.1f}°C (risk={r_th:.2f})")
    if r_tl  > 0.4: notes.append(f"T_low={sensor['T_in']:.1f}°C (risk={r_tl:.2f})")
    if r_h2s > 0.1: notes.append(f"H2S={sensor['H2S_in']:.1f} ppm (risk={r_h2s:.2f})")

    return {
        "risk_level":         level,
        "ventilation_needed": overall >= RISK_TRIGGER_THRESHOLD,
        "rh_risk":            round(r_rh,      3),
        "temp_risk":          round(temp_risk,  3),
        "h2s_risk":           round(r_h2s,     3),
        "overall_score":      round(overall,   3),
        "notes":              " | ".join(notes) if notes else "All readings within safe range",
    }


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 6 — PHYSICS HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def absolute_humidity(T: float, RH: float) -> float:
    """Absolute humidity [kg water / kg dry air] — Buck equation."""
    Psat = 0.61121 * math.exp((18.678 - T / 234.5) * (T / (257.14 + T)))
    return 0.622 * (Psat * RH / (101.325 - Psat * RH))


def air_density(T: float) -> float:
    """Dry air density [kg/m3]."""
    return 353.0 / (T + 273.15)


def latent_heat_of_vaporisation(T: float) -> float:
    """Latent heat [J/g] interpolated from lookup table."""
    temps  = [0,2,4,10,14,18,20,25,30,34,40,44,50,54,60,70,80,90,96]
    latent = [2500.9,2496.4,2491.2,2477.2,2467.7,2458.3,2453.5,2441.7,
              2429.8,2420.3,2406.4,2396.4,2381.9,2372.3,2357.7,2333.7,
              2308.0,2282.5,2266.9]
    if T <= temps[0]:  return latent[0]
    if T >= temps[-1]: return latent[-1]
    for i in range(len(temps) - 1):
        if temps[i] <= T <= temps[i + 1]:
            return latent[i] + (latent[i+1]-latent[i])*(T-temps[i])/(temps[i+1]-temps[i])


def bird_heat_production(W: float, T: float) -> tuple[float, float, float, float]:
    """
    Heat and moisture for one bird.
    W: body weight [kg]  T: indoor temperature [°C]
    Returns: total [W], sensible [W], latent [W], moisture [g/s]
    """
    total    = 10.62 * (W ** 0.75)
    sensible = (0.61 * (1000 + 20*(20 - T) - 0.228*T**2)) * (total / 1000)
    latent   = total - sensible
    moisture = latent / latent_heat_of_vaporisation(T)
    return total, sensible, latent, moisture


def derive_per_bird_params(config: dict) -> dict:
    """Derive q_sensible and m_water from flock_config values."""
    W     = config.get("bird_weight_kg", 2.5)
    T_ref = config.get("T_max", 22.0)
    _, q_sens, _, m_water = bird_heat_production(W, T_ref)
    return {"q_sensible": q_sens, "m_water_per_bird": m_water}


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 7 — SENSOR VALIDATION
# ═══════════════════════════════════════════════════════════════════════════════

def validate_sensors(T_in: float, RH_in: float, H2S_in: float,
                     T_amb: float, RH_amb: float) -> tuple[list, dict]:
    """
    Check each channel against plausibility bounds from SENSOR_RANGES.
    Returns (faults: list[str], valid: dict[str, bool]).
    """
    readings = {"T_in": T_in, "RH_in": RH_in, "H2S_in": H2S_in,
                "T_amb": T_amb, "RH_amb": RH_amb}
    valid  = {ch: SENSOR_RANGES[ch][0] <= v <= SENSOR_RANGES[ch][1]
              for ch, v in readings.items()}
    faults = [ch for ch, ok in valid.items() if not ok]
    return faults, valid


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 8 — DIRECT-INVERSION CONTROL FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════
#hier zou eig ook nog een Co2 rate bij moeten maja we hebben daar geen sensor voor 
def moisture_inversion_rate(prev_rate: float, AH_in: float,
                             AH_target_max: float,
                             AH_outdoor: float) -> tuple[float, bool]:
    """
    Required ventilation [m3/h] derived directly from the RH sensor.

    VR_new = VR_prev * (AH_in - AH_outdoor) / (AH_target_max - AH_outdoor)

    Returns (rate, impossible) where impossible=True means outdoor air is
    already more humid than the indoor target — ventilation cannot help.
    """
    num = AH_in         - AH_outdoor
    den = AH_target_max - AH_outdoor
    if den <= 0: return 0.0, True
    if num <= 0: return 0.0, False
    return prev_rate * (num / den), False


def temperature_inversion_rate(n_birds: int, q_sensible: float,
                                T_target: float, T_amb: float) -> float:
    """
    Required ventilation [m3/h] to hold indoor temperature at T_target.

    VR = Q_sensible_total / (rho * cp * (T_target - T_amb))

    Returns 0 if outside air is already warmer than target
    (ventilation would make things worse, not better).
    """
    delta_T = T_target - T_amb
    if delta_T <= 0: return 0.0
    rho = air_density(T_amb)
    cp  = 1005.0   # J / (kg·K)
    return (n_birds * q_sensible / (rho * cp * delta_T)) * 3600


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 9 — MAIN CONTROL FUNCTION
# ═══════════════════════════════════════════════════════════════════════════════

def compute_ventilation_rate(
    # Sensor inputs (internal names — already normalised by the fetch functions)
    T_in: float, RH_in: float, H2S_in: float,
    T_amb: float, RH_amb: float,
    # Setpoints
    T_min: float, T_max: float, RH_max: float,
    H2S_warning: float, H2S_emergency: float,
    # Flock  (n_birds from cv_counts_colson, params derived from flock_config)
    n_birds: int, q_sensible: float, m_water_per_bird: float,
    # Controller state — pass the dict returned by the previous call
    prev_rate: float, prev_valid: dict, initialised: bool,
    # Hardware limits
    vent_min: float = VENT_MIN_FALLBACK,
    vent_max: float = VENT_MAX_FALLBACK,
    max_slew: float = 50.0,
) -> tuple[float, dict, dict]:
    """
    Compute the ventilation rate for one 30-minute cycle.

    Priority hierarchy
    ------------------
      1. H2S emergency → slam fan to max immediately
      2. H2S warning   → proportional boost on top of other needs
      3. Heat stress   → increase fan if outside is cooler than inside
      4. Moisture      → increase fan if outside is drier than inside
      5. Cold stress   → reduce fan to vent_min

    Returns
    -------
    rate        float — commanded ventilation rate [m3/h]
    new_state   dict  — {prev_rate, prev_valid, initialised} for next call
    diagnostics dict  — what drove the decision (logged + written to Supabase)
    """
    notes  = []

    # ── Validate sensors; substitute last-good values on any fault ────
    faults, valid = validate_sensors(T_in, RH_in, H2S_in, T_amb, RH_amb)
    readings = {"T_in": T_in, "RH_in": RH_in, "H2S_in": H2S_in,
                "T_amb": T_amb, "RH_amb": RH_amb}
    for ch in faults:
        if ch in prev_valid:
            readings[ch] = prev_valid[ch]
            notes.append(f"{ch} faulty — using last known value {prev_valid[ch]:.3f}")
        else:
            notes.append(f"{ch} faulty — no previous value, using raw reading")

    new_prev_valid = {**prev_valid,
                      **{ch: readings[ch] for ch, ok in valid.items() if ok}}

    T_in   = readings["T_in"];   RH_in  = readings["RH_in"]
    H2S_in = readings["H2S_in"]; T_amb  = readings["T_amb"]
    RH_amb = readings["RH_amb"]

    # ── Cold start: seed at vent_max/4 (conservative) ─────────────────
    cold_start = False
    if not initialised:
        prev_rate  = vent_max / 4.0
        cold_start = True
        notes.append(f"Cold start: seeded at {prev_rate:.0f} m3/h (vent_max / 4)")

    # ── Precompute absolute humidity values ───────────────────────────
    AH_in      = absolute_humidity(T_in,  RH_in)
    AH_out     = absolute_humidity(T_amb, RH_amb)
    AH_tgt_max = absolute_humidity(T_in,  RH_max)

    # ── PRIORITY 1: H2S emergency — immediate full-speed override ─────
    if H2S_in >= H2S_emergency:
        rate  = vent_max
        state = dict(prev_rate=rate, prev_valid=new_prev_valid, initialised=True)
        diag  = dict(limiting="H2S emergency", h2s_alert="emergency",
                     vr_moisture=0.0, vr_temp=0.0, rh_impossible=False,
                     cold_start=cold_start, sensor_faults=faults,
                     notes=notes + ["H2S EMERGENCY — fan at maximum"])
        return rate, state, diag

    # ── PRIORITY 2: H2S warning ────────────────────────────────────────
    h2s_alert = "none"
    h2s_boost = 0.0
    if H2S_in >= H2S_warning:
        h2s_alert = "warning"
        h2s_boost = vent_max * 0.15 * (H2S_in - H2S_warning) / max(H2S_warning, 1e-9)
        notes.append(f"H2S warning ({H2S_in:.1f} ppm) — boost +{h2s_boost:.0f} m3/h")

    # ── PRIORITY 3: Heat stress ────────────────────────────────────────
    vr_temp = 0.0
    if T_in > T_max:
        if T_amb < T_in:
            vr_temp = temperature_inversion_rate(n_birds, q_sensible, T_max, T_amb)
            notes.append(f"Heat stress: T_in={T_in:.1f}°C — ventilating to cool")
        else:
            notes.append("Heat stress: outside warmer than inside — fan cannot cool")

    # ── PRIORITY 4: Moisture ───────────────────────────────────────────
    vr_moisture   = 0.0
    rh_impossible = False
    if RH_in > RH_max:
        vr_moisture, rh_impossible = moisture_inversion_rate(
            prev_rate, AH_in, AH_tgt_max, AH_out)
        if rh_impossible:
            notes.append("RH target unachievable — outdoor air too moist")
            vr_moisture = 0.0
        else:
            notes.append(
                f"Excess moisture: RH_in={RH_in*100:.0f}% > RH_max={RH_max*100:.0f}%")

    # ── Combine: highest need wins, apply H2S boost on top ────────────
    if vr_temp >= vr_moisture and vr_temp > 0:
        limiting = "Heat stress"
        target   = vr_temp
    elif vr_moisture > 0:
        limiting = "Moisture"
        target   = vr_moisture
    else:
        limiting = "Baseline"
        target   = vent_min

    target = min(target + h2s_boost, vent_max)
    if h2s_alert == "warning":
        limiting = f"H2S warning + {limiting}"

    # ── PRIORITY 5: Cold stress — reduce to vent_min ──────────────────
    if T_in < T_min:
        target   = vent_min
        limiting = "Cold stress (fan at minimum)"
        notes.append(
            f"Cold stress: T_in={T_in:.1f}°C < T_min={T_min:.1f}°C — fan at minimum")

    # ── Slew rate limit ────────────────────────────────────────────────
    delta = max(-max_slew, min(target - prev_rate, max_slew))
    rate  = max(vent_min, min(prev_rate + delta, vent_max))

    state = dict(prev_rate=rate, prev_valid=new_prev_valid, initialised=True)
    diag  = dict(
        limiting      = limiting,
        h2s_alert     = h2s_alert,
        vr_moisture   = round(vr_moisture, 1),
        vr_temp       = round(vr_temp, 1),
        rh_impossible = rh_impossible,
        cold_start    = cold_start,
        sensor_faults = faults,
        notes         = notes,
    )
    return rate, state, diag


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 10 — CONTROLLER STATE  (persisted between 30-min cron runs)
# ═══════════════════════════════════════════════════════════════════════════════

def load_state() -> dict:
    """Load controller state from disk, or return cold-start defaults."""
    if STATE_FILE.exists():
        try:
            with STATE_FILE.open() as f:
                state = json.load(f)
            log.debug("State loaded: prev_rate=%.0f  initialised=%s",
                      state.get("prev_rate", 0), state.get("initialised"))
            return state
        except (json.JSONDecodeError, KeyError) as exc:
            log.warning("State file corrupt (%s) — cold start", exc)
    return {"prev_rate": 0.0, "prev_valid": {}, "initialised": False}


def save_state(state: dict) -> None:
    """Persist controller state to disk for the next cycle."""
    with STATE_FILE.open("w") as f:
        json.dump(state, f, indent=2)




# ═══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    run_ventilation_cycle()


# ═══════════════════════════════════════════════════════════════════════════════
# APPENDIX — SQL for tables WRITTEN by this script
# Run these once in the Supabase SQL editor.
# Do NOT recreate sensor_readings_colson or cv_counts_colson — they already exist.
# ═══════════════════════════════════════════════════════════════════════════════
#
# -- Setpoints and hardware limits (edit rows in the Supabase dashboard)
# create table flock_config (
#   id             bigserial primary key,
#   active         boolean not null default true,
#   bird_weight_kg float   default 2.5,
#   T_min          float   default 16,      -- °C
#   T_max          float   default 22,      -- °C
#   RH_max         float   default 0.70,    -- fraction 0-1
#   H2S_warning    float   default 1.0,     -- ppm
#   H2S_emergency  float   default 5.0,     -- ppm
#   vent_max_m3h   float   default 150,     -- confirmed fan maximum
#   vent_min_m3h   float   default 0,       -- no minimum yet
#   max_slew_m3h   float   default 50       -- max rate change per 30-min cycle
# );
# -- Insert your first config row:
# insert into flock_config (active) values (true);
#
# -- Risk assessments — written every cycle
# create table risk_assessments (
#   id                 bigserial    primary key,
#   created_at         timestamptz  default now(),
#   risk_level         text         not null,
#   ventilation_needed boolean      not null,
#   rh_risk            float,
#   temp_risk          float,
#   h2s_risk           float,
#   overall_score      float,
#   notes              text
# );
#
# -- Ventilation log — written on each activation
# create table ventilation_log (
#   id              bigserial    primary key,
#   created_at      timestamptz  default now(),
#   rate_m3h        float,
#   limiting_factor text,
#   vr_moisture     float,
#   vr_temp         float,
#   h2s_alert       text,
#   sensor_faults   text,
#   notes           text,
#   risk_level      text,
#   triggered_by    text,
#   n_birds         int,
#   T_amb           float,
#   RH_amb_pct      float
# );
#
# -- Weather cache — one row, upserted each cycle
# create table weather_cache (
#   id          int          primary key default 1,
#   fetched_at  timestamptz,
#   T_amb       float,
#   RH_amb      float,
#   description text,
#   source      text
# );
#
# -- Cron line (every 30 minutes):
# -- */30 * * * *  cd /opt/coop && python ventilation_system.py >> logs/vent.log 2>&1