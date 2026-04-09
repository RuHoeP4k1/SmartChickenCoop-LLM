import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
SCRIPTS_DIR = SCRIPT_DIR / "scripts"

if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from seed_simulation_scenarios import (
    SENSOR_TABLE,
    CV_TABLE,
    build_scenarios,
    build_sensor_rows,
    delete_existing_rows as shared_delete_existing_rows,
    get_supabase_client,
    seed_cv_row,
)

RISK_SNAPSHOT_TABLE = "risk_snapshots_simulation"
CONTROL_CYCLES = 12
FAN_STATE_FILE = SCRIPT_DIR / "vent_state.json"
MOLD_STATE_FILE = SCRIPT_DIR / "mold_state.json"
INITIAL_FAN_RATE = 10.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a simulation scenario step by step over 12 control cycles."
    )
    parser.add_argument(
        "scenario",
        nargs="?",
        help="Scenario to run. Use --list to show available scenarios.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="Show available scenario names and exit.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would happen without Supabase writes or control runs.",
    )
    parser.add_argument(
        "--reset-fan-state",
        action="store_true",
        help="Remove vent_state.json before running.",
    )
    parser.add_argument(
        "--reset-snapshots",
        action="store_true",
        help="Delete all rows from risk_snapshots_simulation before running.",
    )
    parser.add_argument(
        "--reset-all",
        action="store_true",
        help="Apply both --reset-fan-state and --reset-snapshots.",
    )
    return parser.parse_args()


def require_supabase_env() -> None:
    if os.getenv("SUPABASE_URL") and os.getenv("SUPABASE_KEY"):
        return
    raise SystemExit(
        "Missing Supabase credentials. Set SUPABASE_URL and SUPABASE_KEY."
    )


def configure_simulation_tables() -> None:
    os.environ["SENSOR_TABLE"] = SENSOR_TABLE
    os.environ["CV_COUNT_TABLE"] = CV_TABLE
    os.environ["RISK_SNAPSHOT_TABLE"] = RISK_SNAPSHOT_TABLE


def delete_existing_rows(client: Any, table_name: str, scenario_name: str) -> int:
    return shared_delete_existing_rows(client, table_name, scenario_name)


def replace_sensor_rows_up_to_cycle(
    client: Any,
    scenario_name: str,
    sensor_rows: list[dict[str, Any]],
    cycle_index: int,
) -> int:
    visible_rows = sensor_rows[: cycle_index + 1]
    delete_existing_rows(client, SENSOR_TABLE, scenario_name)
    response = client.table(SENSOR_TABLE).insert(visible_rows).execute()
    return len(response.data or [])


def reset_fan_state_file() -> None:
    FAN_STATE_FILE.write_text(
        json.dumps({"prev_rate": INITIAL_FAN_RATE, "initialised": True}, indent=2)
    )


def remove_mold_state_file() -> bool:
    if not MOLD_STATE_FILE.exists():
        return False
    MOLD_STATE_FILE.unlink()
    return True


def clear_table(client: Any, table_name: str) -> int:
    response = client.table(table_name).delete().gte("id", 0).execute()
    return len(response.data or [])


def clear_risk_snapshots(client: Any) -> int:
    return clear_table(client, RISK_SNAPSHOT_TABLE)


def run_control_cycle() -> None:
    from smart_coop_control import main as control_main

    control_main()


def main() -> None:
    args = parse_args()
    scenarios = build_scenarios()

    if args.list:
        print("Available scenarios:")
        for name in sorted(scenarios):
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

    # Every scenario run starts from a clean simulation state so earlier runs
    # cannot leak history into the 12-cycle replay.
    reset_fan_state = True
    reset_snapshots = True

    sensor_rows = build_sensor_rows(scenario_name, scenarios[scenario_name])

    if args.dry_run:
        print(
            f"[dry-run] Would reset {FAN_STATE_FILE.name} to "
            f"prev_rate={INITIAL_FAN_RATE:.0f}."
        )
        print(f"[dry-run] Would remove {MOLD_STATE_FILE.name} if it exists.")
        print(f"[dry-run] Would clear {SENSOR_TABLE}.")
        print(f"[dry-run] Would clear {CV_TABLE}.")
        print(f"[dry-run] Would clear {RISK_SNAPSHOT_TABLE}.")

        for cycle_index in range(len(sensor_rows)):
            current_row = sensor_rows[cycle_index]
            cycle_number = cycle_index + 1
            print(
                f"Running cycle {cycle_number}/{len(sensor_rows)} for scenario {scenario_name}"
            )
            print(
                f"[dry-run] Would replace {SENSOR_TABLE} with rows 0..{cycle_index} "
                f"for scenario {scenario_name}."
            )
            print(
                f"[dry-run] Would replace {CV_TABLE} with timestamp "
                f"{current_row['timestamp']} and cycle_index {current_row['cycle_index']}."
            )
            print("[dry-run] Would execute smart_coop_control.main().")

        print(f"Scenario: {scenario_name}")
        print(f"Cycles executed: {len(sensor_rows)}")
        print(
            "Reset used: "
            f"fan_state={reset_fan_state}, snapshots={reset_snapshots}"
        )
        print("Dry-run: True")
        return

    require_supabase_env()
    configure_simulation_tables()
    client = get_supabase_client()

    reset_fan_state_file()
    print(f"Fan state reset: seeded vent_state.json with {INITIAL_FAN_RATE:.0f} m3/h")

    mold_state_removed = remove_mold_state_file()
    print(
        "Mold state reset: "
        f"{'removed mold_state.json' if mold_state_removed else 'file not present'}"
    )

    sensor_deleted = clear_table(client, SENSOR_TABLE)
    print(f"Simulation sensor rows cleared: {sensor_deleted}")

    cv_deleted = clear_table(client, CV_TABLE)
    print(f"Simulation CV rows cleared: {cv_deleted}")

    deleted = clear_risk_snapshots(client)
    print(f"Risk snapshots cleared: {deleted}")

    for cycle_index in range(len(sensor_rows)):
        cycle_number = cycle_index + 1
        current_row = sensor_rows[cycle_index]
        print(
            f"Running cycle {cycle_number}/{len(sensor_rows)} for scenario {scenario_name}"
        )
        replace_sensor_rows_up_to_cycle(client, scenario_name, sensor_rows, cycle_index)
        seed_cv_row(
            client,
            scenario_name,
            latest_timestamp=current_row["timestamp"],
            cycle_index=current_row["cycle_index"],
        )
        run_control_cycle()

    print(f"Scenario: {scenario_name}")
    print(f"Cycles executed: {len(sensor_rows)}")
    print(
        "Reset used: "
        f"fan_state={reset_fan_state}, snapshots={reset_snapshots}"
    )
    print("Dry-run: False")


if __name__ == "__main__":
    main()
