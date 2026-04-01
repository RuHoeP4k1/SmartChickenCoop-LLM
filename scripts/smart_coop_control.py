import sys
from pathlib import Path
from typing import Any, Dict, List

from supabase import Client

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = Path(__file__).resolve().parent.parent
BACKEND_DIR = ROOT_DIR / "backend"

if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from ventilation_automation import (
    Fan,
    check_connection,
    compute_fan_rate,
    connect,
    load_state,
    log,
    read_bird_count,
    read_weather,
    save_state,
)
from risk_calculation import (
    compute_current_heat_risk_from_readings,
    compute_current_mold_risk_from_state,
)

SENSOR_TABLE = "sensor_readings_colson"
RISK_SNAPSHOT_TABLE = "risk_snapshots"
CV_COUNT_TABLE = "cv_counts_colson"

RECENT_READING_LIMIT = 12
CONTROL_INTERVAL_MINUTES = 10
THI_THRESHOLD = 25.0

Sensors = Dict[str, Any]
Reading = Dict[str, Any]


def fetch_recent_sensor_readings(
    client: Client,
    limit: int = RECENT_READING_LIMIT,
) -> List[Reading]:
    """Fetch recent raw sensor rows for the current control cycle."""
    response = (
        client.table(SENSOR_TABLE)
        .select("*")
        .order("timestamp", desc=True)
        .limit(limit)
        .execute()
    )
    return list(response.data or [])


def build_sensors_from_reading(reading: Reading) -> Sensors:
    """
    Map one raw sensor row to the structure expected by compute_fan_rate().

    TODO: This mapping is intentionally duplicated from
    ventilation_automation.read_sensors(); keep both aligned until sensor
    fetching is shared after hardware tests.
    """
    co2_raw = reading.get("co2_ppm")
    return {
        "T_in": float(reading["temperature_c"]),
        "RH_in": float(reading["humidity_pct"]) / 100.0,
        "H2S_in": float(reading.get("h2s_ppm") or 0.0),
        "CO2_in": float(co2_raw) if co2_raw is not None else None,
    }


def log_sensors(sensors: Sensors) -> None:
    """Log the current indoor sensor snapshot."""
    log.info(
        "Sensors   T=%.1f C  RH=%.0f%%  H2S=%.2f ppm  CO2=%s",
        sensors["T_in"],
        sensors["RH_in"] * 100,
        sensors["H2S_in"],
        f"{sensors['CO2_in']:.0f} ppm"
        if sensors["CO2_in"] is not None
        else "no sensor yet",
    )


def read_previous_mold_state(client: Client) -> Dict[str, Any]:
    """Read the latest persisted mold state from the append-only snapshot table."""
    response = (
        client.table(RISK_SNAPSHOT_TABLE)
        .select("mold_index_m,mold_consecutive_unfavourable_minutes")
        .order("timestamp", desc=True)
        .limit(1)
        .execute()
    )
    if not response.data:
        return {
            "previous_m": 0.0,
            "previous_consecutive_unfavourable_minutes": 0,
        }

    row = response.data[0]
    previous_m = row.get("mold_index_m")
    previous_minutes = row.get("mold_consecutive_unfavourable_minutes")
    if previous_m is None or previous_minutes is None:
        return {
            "previous_m": 0.0,
            "previous_consecutive_unfavourable_minutes": 0,
        }

    return {
        "previous_m": float(previous_m),
        "previous_consecutive_unfavourable_minutes": int(previous_minutes),
    }


def write_risk_snapshot(client: Client, heat_risk: dict, mold_risk: dict) -> None:
    """Insert one new append-only heat and mold snapshot row."""
    payload = {
        "heat_risk_score": heat_risk["risk_score"],
        "heat_risk_level": heat_risk["risk_level"],
        "thi_current": heat_risk["thi"],
        "high_thi_streak_minutes": heat_risk.get("thi_streak_minutes"),
        "mold_risk_score": mold_risk["mold_index_m"],
        "mold_risk_level": mold_risk["mold_risk_level"],
        "mold_favourable_for_growth": mold_risk["mold_favourable_for_growth"],
        "mold_index_m": mold_risk["mold_index_m"],
        "mold_consecutive_unfavourable_minutes": mold_risk[
            "mold_consecutive_unfavourable_minutes"
        ],
        "mold_dmdt_per_24h": mold_risk["mold_dmdt_per_24h"],
        "mold_rhcrit": mold_risk["mold_rhcrit"],
        "mold_mmax": mold_risk["mold_mmax"],
    }

    try:
        client.table(RISK_SNAPSHOT_TABLE).insert(payload).execute()
        log.info(
            "Risk snap heat_score=%.1f level=%s THI=%.2f mold_m=%.4f mold_level=%s",
            heat_risk["risk_score"],
            heat_risk["risk_level"],
            heat_risk["thi"],
            mold_risk["mold_index_m"],
            mold_risk["mold_risk_level"],
        )
    except Exception as exc:
        log.error("Risk snapshot write failed: %s", exc)


def main() -> None:
    log.info("==== Smart coop control cycle ====")

    client = connect()
    if not check_connection(client):
        log.error("Aborting - fix Supabase connection.")
        return

    readings = fetch_recent_sensor_readings(client, limit=RECENT_READING_LIMIT)
    if not readings:
        log.error("Sensor read failed: %s is empty.", SENSOR_TABLE)
        return

    sensors = build_sensors_from_reading(readings[0])
    log_sensors(sensors)

    n_birds = read_bird_count(client)
    T_amb, RH_amb = read_weather()

    try:
        heat_risk = compute_current_heat_risk_from_readings(
            readings=readings,
            thi_threshold=THI_THRESHOLD,
            interval_minutes=CONTROL_INTERVAL_MINUTES,
        )
        log.info(
            "Heat risk score=%.1f level=%s THI=%.2f streak=%s min",
            heat_risk["risk_score"],
            heat_risk["risk_level"],
            heat_risk["thi"],
            heat_risk.get("thi_streak_minutes"),
        )
    except Exception as exc:
        log.error("Heat risk calculation failed: %s", exc)
        return

    previous_mold_state = read_previous_mold_state(client)

    try:
        mold_risk = compute_current_mold_risk_from_state(
            temperature_c=sensors["T_in"],
            humidity_pct=sensors["RH_in"] * 100.0,
            previous_m=previous_mold_state["previous_m"],
            previous_consecutive_unfavourable_minutes=previous_mold_state[
                "previous_consecutive_unfavourable_minutes"
            ],
            sample_minutes=CONTROL_INTERVAL_MINUTES,
        )
        log.info(
            "Mold risk M=%.4f level=%s favourable=%s unfav_min=%d",
            mold_risk["mold_index_m"],
            mold_risk["mold_risk_level"],
            mold_risk["mold_favourable_for_growth"],
            mold_risk["mold_consecutive_unfavourable_minutes"],
        )
    except Exception as exc:
        log.error("Mold risk calculation failed: %s", exc)
        return

    write_risk_snapshot(client, heat_risk, mold_risk)

    prev_rate, initialised = load_state()
    rate, reason = compute_fan_rate(
        sensors=sensors,
        heat_risk=heat_risk,
        T_amb=T_amb,
        RH_amb=RH_amb,
        n_birds=n_birds,
        prev_rate=prev_rate,
        initialised=initialised,
    )

    log.info("Result    rate=%.0f m3/h", rate)
    log.info("Reason    %s", reason)

    Fan().set_rate(rate)
    save_state(rate)

    log.info("==== Smart coop control cycle complete ====")


if __name__ == "__main__":
    main()
