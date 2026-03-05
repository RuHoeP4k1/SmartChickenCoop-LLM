"""
Generate Demo Sensor Data
Creates realistic sensor readings in PostgreSQL for testing
"""

from datetime import datetime, timedelta
import math
import random
from db_utils import insert_sensor_reading, setup_database


# ---------------------------------------------------------------------------
# Continuous-state generator (same logic as scheduler.py simulation)
# Values drift smoothly; status strings derived from numeric values.
# ---------------------------------------------------------------------------

def _temp_target(hour: int) -> float:
    """Baseline temperature target by time of day (smooth sine curve)."""
    return 22 + 10 * math.sin(math.pi * (hour - 4) / 12)


def _classify_temp(t):
    if t >= 35: return "critical"
    if t >= 28 or t <= 10: return "warning"
    return "normal"

def _classify_humidity(h):
    if h >= 85 or h < 30: return "critical"
    if h >= 75 or h < 40: return "warning"
    return "normal"

def _classify_heat_stress(t, h):
    hi = t + 0.33 * h - 4
    if hi >= 40 or t >= 35: return "critical"
    if hi >= 30 or t >= 28: return "warning"
    return "normal"

def _classify_h2s(ppm):
    if ppm >= 10: return "critical"
    if ppm >= 5: return "warning"
    return "normal"

def _classify_mold(score):
    if score >= 70: return "critical"
    if score >= 40: return "warning"
    return "normal"

def _feeder_status(pct):
    if pct < 15: return "empty"
    if pct < 35: return "low"
    return "full"

def _waterer_status(pct):
    if pct < 15: return "empty"
    if pct < 35: return "low"
    return "full"

def _drift(cur, target, speed, noise):
    return cur + (cur - target) * -speed + random.gauss(0, noise)


def _make_timeline_state(timestamps):
    """
    Walk through timestamps in order, drifting a continuous state.
    Yields one reading dict per timestamp.
    """
    # Initial state
    t0 = timestamps[0]
    hour0 = t0.hour
    temp = _temp_target(hour0) + random.gauss(0, 1)
    hum  = max(40, min(90, 75 - (temp - 22) * 0.8)) + random.gauss(0, 2)
    feeder = random.uniform(80, 100)
    waterer = random.uniform(80, 100)
    h2s = random.uniform(0, 2)
    mold = random.uniform(10, 30)
    inside = random.uniform(8, 12)
    eggs = 0
    event_ticks = 0

    for ts in timestamps:
        hour = ts.hour
        t_target = _temp_target(hour)
        h_target = max(40, min(90, 75 - (temp - 22) * 0.8))

        # Occasional events
        if event_ticks <= 0 and random.random() < 0.04:
            event = random.choice(["h2s_spike", "heat_wave", "resource_drain", "cold_snap"])
            if event == "h2s_spike":
                h2s = random.uniform(12, 22)
            elif event == "heat_wave":
                t_target = random.uniform(36, 39)
            elif event == "resource_drain":
                feeder = max(5, feeder - random.uniform(25, 45))
                waterer = max(5, waterer - random.uniform(20, 35))
            elif event == "cold_snap":
                t_target = random.uniform(5, 12)
            event_ticks = random.randint(2, 5)
        elif event_ticks > 0:
            event_ticks -= 1

        temp   = _drift(temp,   t_target, speed=0.15, noise=0.4)
        hum    = max(30, min(95, _drift(hum, h_target, speed=0.12, noise=1.0)))
        h2s    = max(0, _drift(h2s, 1.0, speed=0.20, noise=0.3))
        mold   = max(0, min(100, _drift(mold, 20 + (hum - 55) * 0.8, speed=0.08, noise=1.5)))

        feeder = max(0, feeder - random.uniform(0.05, 0.3))
        waterer = max(0, waterer - random.uniform(0.05, 0.35))
        if feeder < 10 and random.random() < 0.12:
            feeder = random.uniform(80, 100)
        if waterer < 10 and random.random() < 0.12:
            waterer = random.uniform(80, 100)

        inside_target = 12 if (hour >= 20 or hour < 6) else random.uniform(6, 10)
        inside = max(0, min(14, _drift(inside, inside_target, speed=0.1, noise=0.5)))

        if 8 <= hour <= 14 and random.random() < 0.25:
            eggs = min(eggs + 1, 15)
        elif hour >= 20 and random.random() < 0.08:
            eggs = 0

        ventilation_on = temp >= 28 or h2s >= 5
        door_open = 6 <= hour <= 20 and temp < 35

        yield {
            "timestamp": ts,
            "temperature_c": round(temp, 2),
            "temperature_status": _classify_temp(temp),
            "humidity_pct": round(hum, 2),
            "humidity_status": _classify_humidity(hum),
            "heat_stress_index": _classify_heat_stress(temp, hum),
            "feeder_pct": round(feeder, 1),
            "feeder_status": _feeder_status(feeder),
            "waterer_pct": round(waterer, 1),
            "waterer_status": _waterer_status(waterer),
            "chickens_inside": int(round(inside)),
            "egg_count": eggs,
            "h2s_ppm": round(h2s, 2),
            "h2s_level": _classify_h2s(h2s),
            "mold_risk_score": round(mold, 1),
            "mold_risk_status": _classify_mold(mold),
            "door_open": door_open,
            "ventilation_on": ventilation_on,
        }


def generate_scenario_data(scenario: str = "normal"):
    """
    Legacy helper — returns a single snapshot for the given scenario label.
    Still used by generate_simple_test_data().
    """
    hour_map = {
        "normal": 9, "hot_day": 12, "cold_night": 3,
        "critical": 14, "resource_low": 10,
    }
    hour = hour_map.get(scenario, 10)
    ts = [datetime.now()]
    reading = next(_make_timeline_state(ts))

    # Override numeric values to hit the desired scenario range
    if scenario == "hot_day":
        reading["temperature_c"] = random.uniform(28, 32)
    elif scenario == "critical":
        reading["temperature_c"] = random.uniform(35, 38)
        reading["h2s_ppm"] = random.uniform(10, 22)
        reading["mold_risk_score"] = random.uniform(60, 90)
        reading["feeder_pct"] = random.uniform(5, 20)
        reading["waterer_pct"] = random.uniform(5, 20)
    elif scenario == "cold_night":
        reading["temperature_c"] = random.uniform(8, 14)
    elif scenario == "resource_low":
        reading["feeder_pct"] = random.uniform(5, 20)
        reading["waterer_pct"] = random.uniform(5, 20)

    # Re-derive status fields after override
    t = reading["temperature_c"]
    h = reading["humidity_pct"]
    reading["temperature_status"] = _classify_temp(t)
    reading["humidity_status"] = _classify_humidity(h)
    reading["heat_stress_index"] = _classify_heat_stress(t, h)
    reading["feeder_status"] = _feeder_status(reading["feeder_pct"])
    reading["waterer_status"] = _waterer_status(reading["waterer_pct"])
    reading["h2s_level"] = _classify_h2s(reading["h2s_ppm"])
    reading["mold_risk_status"] = _classify_mold(reading["mold_risk_score"])
    return reading


def _build_timestamps(now, days_back, step_minutes_bulk, last_2h_step=5):
    """Build a sorted list of timestamps for bulk history + dense recent window."""
    timestamps = []
    bulk_start = now - timedelta(days=days_back)
    bulk_end   = now - timedelta(hours=2)
    t = bulk_start
    while t < bulk_end:
        timestamps.append(t)
        t += timedelta(minutes=step_minutes_bulk)
    # Last 2h at 5-min resolution
    for i in range(int(120 / last_2h_step) + 1):
        ts = now - timedelta(hours=2) + timedelta(minutes=i * last_2h_step)
        if ts <= now:
            timestamps.append(ts)
    return timestamps


def generate_24h_timeline():
    """
    Generate a realistic 24-hour timeline of sensor readings.
    Values drift continuously — no hard scenario snaps.
    """
    print("Generating 24-hour sensor data timeline...")
    now = datetime.now()
    timestamps = _build_timestamps(now, days_back=1, step_minutes_bulk=15)
    total = len(timestamps)
    for i, reading in enumerate(_make_timeline_state(timestamps), 1):
        insert_sensor_reading(reading)
        if i % 20 == 0:
            print(f"  Progress: {i}/{total}", end='\r')
    print(f"\n[OK] Added {total} sensor readings")


def generate_7d_timeline():
    """
    Generate a 7-day timeline so the 7d chart has real data.
    Values drift continuously — no hard scenario snaps.
    """
    print("Generating 7-day sensor data timeline...")
    now = datetime.now()
    # Days 7→2 at 30-min, last 2 days at 15-min, last 2h at 5-min
    timestamps = []
    # Days 7→2 at 30-min
    bulk_start = now - timedelta(days=7)
    bulk_end   = now - timedelta(days=2)
    t = bulk_start
    while t < bulk_end:
        timestamps.append(t)
        t += timedelta(minutes=30)
    # Last 2 days at 15-min
    dense_end = now - timedelta(hours=2)
    t = bulk_end
    while t < dense_end:
        timestamps.append(t)
        t += timedelta(minutes=15)
    # Last 2h at 5-min
    for i in range(25):
        ts = now - timedelta(hours=2) + timedelta(minutes=i * 5)
        if ts <= now:
            timestamps.append(ts)

    total = len(timestamps)
    for i, reading in enumerate(_make_timeline_state(timestamps), 1):
        insert_sensor_reading(reading)
        if i % 50 == 0:
            print(f"  Progress: {i}/{total}", end='\r')
    print(f"\n[OK] Added {total} sensor readings")


def generate_simple_test_data():
    """
    Generate just a few readings for quick testing.
    """
    
    print("Generating simple test data (5 readings)...")
    
    test_scenarios = [
        ("normal", datetime.now() - timedelta(hours=2)),
        ("normal", datetime.now() - timedelta(hours=1, minutes=30)),
        ("hot_day", datetime.now() - timedelta(hours=1)),
        ("hot_day", datetime.now() - timedelta(minutes=30)),
        ("critical", datetime.now() - timedelta(minutes=5))
    ]

    for scenario, timestamp in test_scenarios:
        sensor_data = generate_scenario_data(scenario)
        sensor_data['timestamp'] = timestamp
        insert_sensor_reading(sensor_data)
        print(f"  [OK] {scenario.ljust(12)} @ {timestamp.strftime('%H:%M')}")

    print("[OK] Test data ready")


def show_latest_reading():
    """
    Display the most recent sensor reading.
    """
    from db_utils import get_latest_sensor_reading
    
    print("\n" + "="*70)
    print("LATEST SENSOR READING")
    print("="*70)
    
    latest = get_latest_sensor_reading()
    
    if not latest:
        print("No data in database yet!")
        return
    
    print(f"Timestamp: {latest['timestamp']}")
    print(f"Temperature: {latest['temperature_c']:.1f}°C [{latest['temperature_status']}]")
    print(f"Humidity: {latest['humidity_pct']:.1f}% [{latest['humidity_status']}]")
    print(f"Heat Stress: {latest['heat_stress_index']}")
    print(f"Feeder: {latest['feeder_status']} ({latest.get('feeder_pct', 'N/A')}%)")
    print(f"Waterer: {latest['waterer_status']} ({latest.get('waterer_pct', 'N/A')}%)")
    print(f"Chickens inside: {latest.get('chickens_inside', 'N/A')}")
    print(f"Egg count: {latest.get('egg_count', 'N/A')}")
    print(f"H2S: {latest.get('h2s_ppm', 'N/A')} ppm [{latest.get('h2s_level', 'N/A')}]")
    print(f"Mold risk: {latest.get('mold_risk_score', 'N/A')} [{latest.get('mold_risk_status', 'N/A')}]")
    print(f"Door open: {latest.get('door_open', 'N/A')}")
    print(f"Ventilation on: {latest.get('ventilation_on', 'N/A')}")


if __name__ == "__main__":
    
    print("="*70)
    print("DEMO SENSOR DATA GENERATOR")
    print("="*70)
    print()
    
    # Setup database
    print("Step 1: Setting up database...")
    try:
        setup_database()
    except Exception as e:
        print(f"❌ Database setup failed: {e}")
        print("\nMake sure PostgreSQL is running!")
        print("Check DB_CONFIG in db_utils.py")
        exit(1)
    
    print()
    
    # Choose scenario
    print("Choose what to generate:")
    print("  1. Simple test data (5 readings)     — quick sanity check")
    print("  2. 24-hour timeline (~110 readings)  — fills 1h + 24h charts")
    print("  3. 7-day timeline  (~500 readings)   — fills ALL charts (RECOMMENDED)")
    print()

    choice = input("Enter choice (1, 2 or 3): ").strip()
    print()

    if choice == "3":
        generate_7d_timeline()
    elif choice == "2":
        generate_24h_timeline()
    else:
        generate_simple_test_data()
    
    # Show latest
    show_latest_reading()
    
    print("\n" + "="*70)
    print("✅ Demo data generation complete!")
    print()
    print("Next steps:")
    print("  1. Run: python rag_functions.py")
    print("  2. Or run: python evaluate_rag.py")
    print("="*70)
