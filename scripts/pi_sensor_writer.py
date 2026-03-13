"""
Pi Sensor Writer — Template for the sensor team
Copy this file onto the Raspberry Pi. It lets you push sensor readings
to the PostgreSQL database on the desktop without writing any SQL.

Setup on the Pi: (Ruben will help you with this)
    1. pip install psycopg2-binary python-dotenv
    2. Create a .env file (or set env vars) with:
           DB_HOST=<desktop IP address, e.g. 192.168.1.42>
           DB_PORT=5432
           DB_NAME=chickens
           DB_USER=postgres
           DB_PASSWORD=<your password>
    3. Import send_reading() in your sensor script and call it.

Expected field values (you decide what's normal/warning/critical):
    temperature_c        float    e.g. 22.5
    temperature_status   str      "normal" | "warning" | "critical"
    humidity_pct         float    e.g. 65.0
    humidity_status      str      "normal" | "warning" | "critical"
    heat_stress_index    str      "normal" | "warning" | "critical"
    feeder_status        str      "full" | "low" | "empty"
    waterer_status       str      "full" | "low" | "empty"
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
from backend.db_utils import insert_sensor_reading, get_db_connection, release_db_connection

# You can modify this function to adjust for the format of your sensor readings.
# Remind more functions will have to be refactored if you change the expected field names or types.

def send_reading(
    temperature_c: float,
    temperature_status: str,
    humidity_pct: float,
    humidity_status: str,
    heat_stress_index: str = "normal",
    feeder_status: str = "full",
    waterer_status: str = "full",
) -> int:
    """
    Send a sensor reading to the database.

    Returns:
        The ID of the inserted row
    """
    reading = {
        "temperature_c": temperature_c,
        "temperature_status": temperature_status,
        "humidity_pct": humidity_pct,
        "humidity_status": humidity_status,
        "heat_stress_index": heat_stress_index,
        "feeder_status": feeder_status,
        "waterer_status": waterer_status,
    }

    row_id = insert_sensor_reading(reading)
    return row_id


def test_connection() -> bool:
    """Test if the database connection works. Returns True on success."""
    try:
        conn = get_db_connection()
        release_db_connection(conn)
        return True
    except Exception as e:
        print(f"Connection failed: {e}")
        return False


# ---------------------------------------------------------------------------
# Example: periodic write loop
# Replace read_sensors() with your actual sensor code.
# ---------------------------------------------------------------------------

def read_sensors() -> dict:
    """
    PLACEHOLDER — Replace this with your actual sensor reading code.
    Should return a dict with the values send_reading() expects.
    """
    # Example: replace with real DHT22 / BME280 / etc. readings
    return {
        "temperature_c": 22.0,
        "temperature_status": "normal",
        "humidity_pct": 55.0,
        "humidity_status": "normal",
        "heat_stress_index": "normal",
        "feeder_status": "full",
        "waterer_status": "full",
    }

def run_loop(interval_seconds: int = 120):
    """
    Run a periodic sensor write loop.
    Reads sensors every interval_seconds and pushes to the DB.

    Args:
        interval_seconds: How often to write (default 2 minutes)
    """
    print(f"Starting sensor loop (every {interval_seconds}s). Ctrl+C to stop.")

    while True:
        try:
            data = read_sensors()
            row_id = send_reading(**data)
            print(f"  Written row id={row_id} | temp={data['temperature_c']}C [{data['temperature_status']}]")
        except Exception as e:
            print(f"  Write failed: {e}")

        time.sleep(interval_seconds)


if __name__ == "__main__":
    print("Pi Sensor Writer — Connection Test")
    print("=" * 40)

    if not test_connection():
        print("\nCheck your .env or environment variables:")
        print("  DB_HOST  = IP address of the desktop running PostgreSQL")
        print("  DB_PORT  = 5432 (default)")
        print("  DB_NAME  = chickens")
        print("  DB_USER  = postgres")
        print("  DB_PASSWORD = <your password>")
        exit(1)

    print("Connection OK!\n")
    print("Inserting a test reading...")

    row_id = send_reading(
        temperature_c=22.0,
        temperature_status="normal",
        humidity_pct=55.0,
        humidity_status="normal",
        heat_stress_index="normal",
        feeder_status="full",
        waterer_status="full",
    )

    print(f"Inserted row id={row_id}")
    print("\nTo start the periodic loop, call run_loop() or run:")
    print("  python -c \"from pi_sensor_writer import run_loop; run_loop()\"")
