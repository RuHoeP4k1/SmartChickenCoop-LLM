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
            "waterer_status": "full"
        },
        
        "hot_day": {
            "temperature_c": random.uniform(28, 32),
            "temperature_status": "warning",
            "humidity_pct": random.uniform(65, 80),
            "humidity_status": "warning",
            "heat_stress_index": "warning",
            "feeder_status": random.choice(["full", "full", "low"]),
            "waterer_status": random.choice(["full", "low"])
        },
        
        "critical": {
            "temperature_c": random.uniform(35, 38),
            "temperature_status": "critical",
            "humidity_pct": random.uniform(80, 90),
            "humidity_status": "critical",
            "heat_stress_index": "critical",
            "feeder_status": random.choice(["low", "empty"]),
            "waterer_status": random.choice(["low", "empty"])
        },
        
        "cold_night": {
            "temperature_c": random.uniform(8, 14),
            "temperature_status": "warning",
            "humidity_pct": random.uniform(60, 75),
            "humidity_status": "normal",
            "heat_stress_index": "normal",
            "feeder_status": "full",
            "waterer_status": "full"
        },
        
        "resource_low": {
            "temperature_c": random.uniform(20, 24),
            "temperature_status": "normal",
            "humidity_pct": random.uniform(50, 65),
            "humidity_status": "normal",
            "heat_stress_index": "normal",
            "feeder_status": "low",
            "waterer_status": "low"
        }
    }
    
    return scenarios.get(scenario, scenarios["normal"])


def generate_24h_timeline():
    """
    Generate a realistic 24-hour timeline of sensor readings.
    Simulates temperature changes throughout the day.
    """
    
    print("Generating 24-hour sensor data timeline...")
    print("This simulates readings every 15 minutes for one day.")
    print()
    
    readings_added = 0
    base_time = datetime.now() - timedelta(days=1)
    
    # Generate readings every 15 minutes
    for i in range(96):  # 24 hours * 4 readings/hour
        timestamp = base_time + timedelta(minutes=i*15)
        hour = timestamp.hour
        
        # Simulate realistic daily temperature pattern
        if 6 <= hour < 10:
            scenario = "normal"  # Morning
        elif 10 <= hour < 14:
            scenario = "hot_day"  # Midday heat
        elif 14 <= hour < 16:
            scenario = random.choice(["hot_day", "critical"])  # Hottest part
        elif 16 <= hour < 20:
            scenario = "hot_day"  # Evening cooling
        elif 20 <= hour < 22:
            scenario = "normal"  # Night
        else:
            scenario = "cold_night"  # Late night/early morning
        
        # Random resource depletion throughout day
        if random.random() < 0.1:  # 10% chance
            scenario = "resource_low"
        
        sensor_data = generate_scenario_data(scenario)
        sensor_data['timestamp'] = timestamp
        
        insert_sensor_reading(sensor_data)
        readings_added += 1
        
        if readings_added % 20 == 0:
            print(f"  Progress: {readings_added}/96 readings", end='\r')
    
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
    print(f"Feeder: {latest['feeder_status']}")
    print(f"Waterer: {latest['waterer_status']}")


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
    print("  1. Simple test data (5 readings) - RECOMMENDED FOR FIRST RUN")
    print("  2. Full 24-hour timeline (96 readings)")
    print()
    
    choice = input("Enter choice (1 or 2): ").strip()
    print()
    
    if choice == "2":
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
