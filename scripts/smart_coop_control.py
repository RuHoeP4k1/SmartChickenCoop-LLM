import sys
from pathlib import Path

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
    read_sensors,
    read_weather,
    save_state,
)
from risk_calculation import compute_current_heat_risk_from_recent_readings


def write_risk_snapshot(client: Client, heat_risk: dict) -> None:
    """
    Write the current heat risk to risk_snapshots.
    Only heat-related columns are populated for now.
    Mold-related columns remain null.
    """
    payload = {
        "heat_risk_score": heat_risk["risk_score"],
        "heat_risk_level": heat_risk["risk_level"],
        "thi_current": heat_risk["thi"],
        "high_thi_streak_minutes": heat_risk.get("thi_streak_minutes"),
        "mold_risk_score": None,
        "mold_risk_level": None,
        "mold_favourable_for_growth": None,
    }

    try:
        client.table("risk_snapshots").insert(payload).execute()
        log.info(
            "Risk snap heat_score=%.1f level=%s THI=%.2f",
            heat_risk["risk_score"],
            heat_risk["risk_level"],
            heat_risk["thi"],
        )
    except Exception as exc:
        log.error("Risk snapshot write failed: %s", exc)


def main() -> None:
    log.info("════ Smart coop control cycle ════")

    client = connect()
    if not check_connection(client):
        log.error("Aborting — fix Supabase connection.")
        return

    try:
        sensors = read_sensors(client)
    except ValueError as exc:
        log.error("Sensor read failed: %s", exc)
        return

    n_birds = read_bird_count(client)
    T_amb, RH_amb = read_weather()

    try:
        heat_risk = compute_current_heat_risk_from_recent_readings(
            table_name="sensor_readings_colson",
            limit=12,
            thi_threshold=25.0,
            interval_minutes=10,
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

    write_risk_snapshot(client, heat_risk)

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

    log.info("════ Smart coop control cycle complete ════")


if __name__ == "__main__":
    main()
