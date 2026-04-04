import argparse
import os
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List

from supabase import Client, create_client

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

SENSOR_TABLE = "sensor_readings_simulation"
CV_TABLE = "cv_counts_simulation"
CONTROL_CYCLES = 12
CONTROL_INTERVAL_MINUTES = 10

def get_supabase_client() -> Client:
    """Create a Supabase client from environment variables."""
    supabase_url = os.getenv("SUPABASE_URL")
    supabase_key = os.getenv("SUPABASE_KEY")

    if not supabase_url or not supabase_key:
        raise ValueError(
            "Missing Supabase credentials. Set SUPABASE_URL and SUPABASE_KEY."
        )

    return create_client(supabase_url, supabase_key)


def build_scenarios() -> Dict[str, Dict[str, Any]]:
    """Return the simple indoor-only simulation scenarios."""
    return {
        "baseline": {
            "temperature_c": 20.0,
            "humidity_pct": 60.0,
            "co2_ppm": 800.0,
            "h2s_ppm": 0.0,
        },
        "co2_high": {
            "temperature_c": 20.0,
            "humidity_pct": 60.0,
            "co2_ppm": 2600.0,
            "h2s_ppm": 0.0,
        },
        "h2s_warning": {
            "temperature_c": 20.0,
            "humidity_pct": 60.0,
            "co2_ppm": 900.0,
            "h2s_ppm": 2.0,
        },
        "h2s_emergency": {
            "temperature_c": 20.0,
            "humidity_pct": 60.0,
            "co2_ppm": 900.0,
            "h2s_ppm": 5.5,
        },
        "temperature_high": {
            "temperature_c": 30.0,
            "humidity_pct": 60.0,
            "co2_ppm": 900.0,
            "h2s_ppm": 0.0,
        },
        "humidity_high": {
            "temperature_c": 25.0,
            "humidity_pct": 90.0,
            "co2_ppm": 900.0,
            "h2s_ppm": 0.0,
        },
        "thi_streak_warmup": {
            "series": [
                {"temperature_c": 16.0, "humidity_pct": 60.0, "co2_ppm": 900.0, "h2s_ppm": 0.0},
                {"temperature_c": 17.0, "humidity_pct": 60.0, "co2_ppm": 900.0, "h2s_ppm": 0.0},
                {"temperature_c": 18.0, "humidity_pct": 60.0, "co2_ppm": 900.0, "h2s_ppm": 0.0},
                {"temperature_c": 20.0, "humidity_pct": 62.0, "co2_ppm": 900.0, "h2s_ppm": 0.0},
                {"temperature_c": 22.0, "humidity_pct": 64.0, "co2_ppm": 900.0, "h2s_ppm": 0.0},
                {"temperature_c": 24.0, "humidity_pct": 66.0, "co2_ppm": 900.0, "h2s_ppm": 0.0},
                {"temperature_c": 26.0, "humidity_pct": 68.0, "co2_ppm": 900.0, "h2s_ppm": 0.0},
                {"temperature_c": 27.0, "humidity_pct": 70.0, "co2_ppm": 900.0, "h2s_ppm": 0.0},
                {"temperature_c": 28.0, "humidity_pct": 72.0, "co2_ppm": 900.0, "h2s_ppm": 0.0},
                {"temperature_c": 29.0, "humidity_pct": 74.0, "co2_ppm": 900.0, "h2s_ppm": 0.0},
                {"temperature_c": 30.0, "humidity_pct": 76.0, "co2_ppm": 900.0, "h2s_ppm": 0.0},
                {"temperature_c": 31.0, "humidity_pct": 78.0, "co2_ppm": 900.0, "h2s_ppm": 0.0},
            ],
        },
    }


def build_sensor_rows(
    scenario_name: str,
    scenario_values: Dict[str, Any],
    cycle_count: int = CONTROL_CYCLES,
    interval_minutes: int = CONTROL_INTERVAL_MINUTES,
) -> List[Dict[str, Any]]:
    """Build one row per control cycle, ending at the current UTC time."""
    end_time = datetime.now(timezone.utc).replace(microsecond=0)
    start_time = end_time - timedelta(minutes=interval_minutes * (cycle_count - 1))
    notes = f"Simple indoor-only simulation seed for scenario '{scenario_name}'."
    if scenario_name == "thi_streak_warmup":
        notes = "Indoor-only warmup scenario to test THI streak buildup."
    rows: List[Dict[str, Any]] = []

    for cycle_index in range(cycle_count):
        timestamp = start_time + timedelta(minutes=interval_minutes * cycle_index)
        row_values = scenario_values
        if scenario_name == "thi_streak_warmup":
            row_values = scenario_values["series"][cycle_index]
        rows.append(
            {
                "timestamp": timestamp.isoformat().replace("+00:00", "Z"),
                "temperature_c": row_values["temperature_c"],
                "humidity_pct": row_values["humidity_pct"],
                "co2_ppm": row_values["co2_ppm"],
                "h2s_ppm": row_values["h2s_ppm"],
                "scenario_name": scenario_name,
                "cycle_index": cycle_index,
                "notes": notes,
            }
        )

    return rows


def delete_existing_rows(client: Client, table_name: str, scenario_name: str) -> int:
    """Remove older rows for the same scenario before reseeding."""
    response = (
        client.table(table_name)
        .delete()
        .eq("scenario_name", scenario_name)
        .execute()
    )
    return len(response.data or [])


def seed_sensor_rows(
    client: Client,
    scenario_name: str,
    sensor_rows: List[Dict[str, Any]],
) -> int:
    """Replace sensor rows for the chosen scenario."""
    delete_existing_rows(client, SENSOR_TABLE, scenario_name)
    response = client.table(SENSOR_TABLE).insert(sensor_rows).execute()
    return len(response.data or [])


def seed_cv_row(
    client: Client,
    scenario_name: str,
    latest_timestamp: str,
    cycle_index: int,
) -> int:
    """Replace the single CV row for the chosen scenario."""
    delete_existing_rows(client, CV_TABLE, scenario_name)

    payload = {
        "timestamp": latest_timestamp,
        "number_of_chickens": 6,
        "egg_count": 0,
        "scenario_name": scenario_name,
        "cycle_index": cycle_index,
        "notes": f"Bird count seed for simulation scenario '{scenario_name}'.",
    }
    response = client.table(CV_TABLE).insert(payload).execute()
    return len(response.data or [])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Seed simple indoor-only simulation scenarios into Supabase."
    )
    parser.add_argument(
        "scenario",
        nargs="?",
        help="Scenario to seed. Use --list to show available scenarios.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="Show available scenario names and exit.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    scenarios = build_scenarios()

    if args.list:
        print("Available scenarios:")
        for name in scenarios:
            print(f"- {name}")
        return

    if not args.scenario:
        raise SystemExit("Please provide a scenario name or use --list.")

    scenario_name = args.scenario
    if scenario_name not in scenarios:
        available = ", ".join(sorted(scenarios))
        raise SystemExit(
            f"Unknown scenario '{scenario_name}'. Available scenarios: {available}"
        )

    client = get_supabase_client()
    sensor_rows = build_sensor_rows(scenario_name, scenarios[scenario_name])
    sensor_written = seed_sensor_rows(client, scenario_name, sensor_rows)
    cv_written = seed_cv_row(
        client,
        scenario_name,
        latest_timestamp=sensor_rows[-1]["timestamp"],
        cycle_index=sensor_rows[-1]["cycle_index"],
    )

    print(f"Scenario: {scenario_name}")
    print(f"Sensor rows written: {sensor_written}")
    print(f"CV rows written: {cv_written}")


if __name__ == "__main__":
    main()
