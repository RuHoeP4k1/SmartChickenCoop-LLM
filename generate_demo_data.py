"""
Generate Demo Sensor Data
Creates realistic sensor readings in PostgreSQL for testing
"""

from datetime import datetime, timedelta
import random
from db_utils import insert_sensor_reading, setup_database


def generate_scenario_data(scenario: str = "normal"):
    """
    Generate sensor data for different scenarios.
    
    Args:
        scenario: "normal", "hot_day", "cold_night", "critical", "resource_low"
    
    Returns:
        Dictionary with sensor data
    """
    
    scenarios = {
        "normal": {
            "temperature_c": random.uniform(20, 34),
            "temperature_status": "normal",
            "humidity_pct": random.uniform(50, 90),
            "humidity_status": "normal",
            "heat_stress_index": "normal",
            "feeder_status": "full",
            "waterer_status": "full",
            "feeder_pct": random.uniform(70, 100),
            "waterer_pct": random.uniform(70, 100),
            "chickens_inside": random.randint(8, 12),
            "egg_count": random.randint(0, 5),
            "h2s_ppm": random.uniform(0, 2),
            "h2s_level": "normal",
            "mold_risk_score": random.uniform(0, 20),
            "mold_risk_status": "normal",
            "door_open": False,
            "ventilation_on": False,
        },

        "hot_day": {
            "temperature_c": random.uniform(28, 32),
            "temperature_status": "warning",
            "humidity_pct": random.uniform(65, 80),
            "humidity_status": "warning",
            "heat_stress_index": "warning",
            "feeder_status": random.choice(["full", "full", "low"]),
            "waterer_status": random.choice(["full", "low"]),
            "feeder_pct": random.uniform(40, 80),
            "waterer_pct": random.uniform(30, 70),
            "chickens_inside": random.randint(5, 10),
            "egg_count": random.randint(0, 3),
            "h2s_ppm": random.uniform(1, 5),
            "h2s_level": "normal",
            "mold_risk_score": random.uniform(20, 50),
            "mold_risk_status": random.choice(["normal", "warning"]),
            "door_open": True,
            "ventilation_on": True,
        },

        "critical": {
            "temperature_c": random.uniform(35, 38),
            "temperature_status": "critical",
            "humidity_pct": random.uniform(80, 90),
            "humidity_status": "critical",
            "heat_stress_index": "critical",
            "feeder_status": random.choice(["low", "empty"]),
            "waterer_status": random.choice(["low", "empty"]),
            "feeder_pct": random.uniform(5, 25),
            "waterer_pct": random.uniform(5, 20),
            "chickens_inside": random.randint(2, 8),
            "egg_count": 0,
            "h2s_ppm": random.uniform(10, 25),
            "h2s_level": random.choice(["warning", "critical"]),
            "mold_risk_score": random.uniform(60, 95),
            "mold_risk_status": "critical",
            "door_open": random.choice([True, False]),
            "ventilation_on": False,
        },

        "cold_night": {
            "temperature_c": random.uniform(8, 14),
            "temperature_status": "warning",
            "humidity_pct": random.uniform(60, 75),
            "humidity_status": "normal",
            "heat_stress_index": "normal",
            "feeder_status": "full",
            "waterer_status": "full",
            "feeder_pct": random.uniform(60, 100),
            "waterer_pct": random.uniform(60, 100),
            "chickens_inside": random.randint(10, 14),
            "egg_count": random.randint(0, 8),
            "h2s_ppm": random.uniform(0, 3),
            "h2s_level": "normal",
            "mold_risk_score": random.uniform(30, 60),
            "mold_risk_status": random.choice(["normal", "warning"]),
            "door_open": False,
            "ventilation_on": random.choice([True, False]),
        },

        "resource_low": {
            "temperature_c": random.uniform(20, 24),
            "temperature_status": "normal",
            "humidity_pct": random.uniform(50, 65),
            "humidity_status": "normal",
            "heat_stress_index": "normal",
            "feeder_status": "low",
            "waterer_status": "low",
            "feeder_pct": random.uniform(5, 20),
            "waterer_pct": random.uniform(5, 20),
            "chickens_inside": random.randint(8, 12),
            "egg_count": random.randint(0, 4),
            "h2s_ppm": random.uniform(0, 3),
            "h2s_level": "normal",
            "mold_risk_score": random.uniform(10, 35),
            "mold_risk_status": "normal",
            "door_open": False,
            "ventilation_on": False,
        }
    }
    
    return scenarios.get(scenario, scenarios["normal"])


def _scenario_for_hour(hour):
    """Pick a scenario based on time of day."""
    if 6 <= hour < 10:
        scenario = "normal"
    elif 10 <= hour < 14:
        scenario = "hot_day"
    elif 14 <= hour < 16:
        scenario = random.choice(["hot_day", "critical"])
    elif 16 <= hour < 20:
        scenario = "hot_day"
    elif 20 <= hour < 22:
        scenario = "normal"
    else:
        scenario = "cold_night"
    if random.random() < 0.1:
        scenario = "resource_low"
    return scenario


def generate_24h_timeline():
    """
    Generate a realistic 24-hour timeline of sensor readings.
    15-minute intervals for 24h + 5-minute intervals for the last 2h
    so the 1h chart always has plenty of points.
    """

    print("Generating 24-hour sensor data timeline...")
    print()

    readings_added = 0
    now = datetime.now()

    # 24h ago → 2h ago at 15-min resolution
    base_time = now - timedelta(days=1)
    steps_15m = int((22 * 60) / 15)   # 22 hours worth
    total = steps_15m + 24             # + 24 readings at 5-min for last 2h
    for i in range(steps_15m):
        timestamp = base_time + timedelta(minutes=i * 15)
        sensor_data = generate_scenario_data(_scenario_for_hour(timestamp.hour))
        sensor_data['timestamp'] = timestamp
        insert_sensor_reading(sensor_data)
        readings_added += 1
        if readings_added % 20 == 0:
            print(f"  Progress: {readings_added}/{total}", end='\r')

    # Last 2h at 5-min resolution (gives ~24 points in the 1h chart)
    for i in range(24):
        timestamp = now - timedelta(hours=2) + timedelta(minutes=i * 5)
        sensor_data = generate_scenario_data(_scenario_for_hour(timestamp.hour))
        sensor_data['timestamp'] = timestamp
        insert_sensor_reading(sensor_data)
        readings_added += 1

    print(f"\n✅ Added {readings_added} sensor readings")


def generate_7d_timeline():
    """
    Generate a 7-day timeline so the 7d chart has real data.
    15-minute intervals for days 7–2 + denser data for the last 2 days.
    """

    print("Generating 7-day sensor data timeline...")
    print()

    readings_added = 0
    now = datetime.now()

    # Days 7 → 2: one reading every 30 min
    base_time = now - timedelta(days=7)
    steps_30m = int((5 * 24 * 60) / 30)   # 5 days
    for i in range(steps_30m):
        timestamp = base_time + timedelta(minutes=i * 30)
        sensor_data = generate_scenario_data(_scenario_for_hour(timestamp.hour))
        sensor_data['timestamp'] = timestamp
        insert_sensor_reading(sensor_data)
        readings_added += 1
        if readings_added % 50 == 0:
            print(f"  Progress (bulk): {readings_added}", end='\r')

    print()

    # Last 2 days at 15-min resolution
    base_time = now - timedelta(days=2)
    steps_15m = int((2 * 24 * 60) / 15)
    for i in range(steps_15m):
        timestamp = base_time + timedelta(minutes=i * 15)
        sensor_data = generate_scenario_data(_scenario_for_hour(timestamp.hour))
        sensor_data['timestamp'] = timestamp
        insert_sensor_reading(sensor_data)
        readings_added += 1

    # Last 2h at 5-min resolution
    for i in range(24):
        timestamp = now - timedelta(hours=2) + timedelta(minutes=i * 5)
        sensor_data = generate_scenario_data(_scenario_for_hour(timestamp.hour))
        sensor_data['timestamp'] = timestamp
        insert_sensor_reading(sensor_data)
        readings_added += 1

    print(f"\n✅ Added {readings_added} sensor readings")


def generate_simple_test_data():
    """
    Generate just a few readings for quick testing.
    """
    
    print("Generating simple test data (5 readings)...")
    
    test_scenarios = [
        ("normal", datetime.now() - timedelta(hours=2)),
        ("normal", datetime.now() - timedelta(hours=1, minutes=30)),
        ("hot_day", datetime.now() - timedelta(hours=1)),
        ("warning", datetime.now() - timedelta(minutes=30)),
        ("critical", datetime.now() - timedelta(minutes=5))
    ]
    
    for scenario, timestamp in test_scenarios:
        sensor_data = generate_scenario_data(scenario)
        sensor_data['timestamp'] = timestamp
        insert_sensor_reading(sensor_data)
        print(f"  ✓ {scenario.ljust(12)} @ {timestamp.strftime('%H:%M')}")
    
    print("✅ Test data ready")


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
