"""
Smart sensor context filtering
Decides when to include sensor data in LLM prompt to reduce token usage
"""

from datetime import datetime, timedelta, timezone
from typing import Dict, Optional

# If the latest reading is older than this, treat it as stale and ignore it.
# 30 minutes is generous — the Pi should write every ~2 minutes.
STALE_THRESHOLD = timedelta(minutes=30)


def is_reading_stale(sensor_data: Dict) -> bool:
    """
    Check whether a sensor reading is too old to be trusted.

    Returns True if the reading's timestamp is older than STALE_THRESHOLD,
    or if there is no timestamp at all.
    """
    ts = sensor_data.get("timestamp")
    if ts is None:
        return True

    # psycopg2 returns datetime objects; handle both naive and aware
    if isinstance(ts, str):
        ts = datetime.fromisoformat(ts)

    now = datetime.now(ts.tzinfo)  # match tz-awareness of the DB value
    return (now - ts) > STALE_THRESHOLD


def is_environment_query(user_query: str) -> bool:
    """
    Check if the query is specifically about coop conditions or distress symptoms.

    This is separate from critical sensor checks — it only looks at the query text.
    Used to decide whether emergency mode is appropriate.
    """
    environment_keywords = [
        # Direct coop condition questions
        "temperature in", "temp in", "how hot is", "how cold is",
        "coop temperature", "coop temp",
        "humidity in", "humidity level",

        # Active distress symptoms (likely environment-caused)
        "panting", "heat stress", "heat stroke", "overheating",
        "freezing", "hypothermia",
        "lethargic", "not moving", "breathing heavy",
        "wings spread",

        # Direct sensor/coop references
        "feeder", "waterer", "sensor", "coop conditions",
        "what are the readings", "check the coop"
    ]

    query_lower = user_query.lower()
    return any(keyword in query_lower for keyword in environment_keywords)


def should_include_sensors(user_query: str, sensor_data: Dict) -> bool:
    """
    Decide if we need to include sensor data in the LLM prompt.

    Strategy:
    1. If query is about environment/distress → include
    2. If query is about resources AND resources are low → include
    3. If critical alerts exist → include as brief note (NOT emergency mode)
    4. General questions → skip sensors

    Args:
        user_query: User's question
        sensor_data: Current sensor readings from database

    Returns:
        True if sensor context should be included
    """

    if not sensor_data:
        return False

    # Stale readings are unreliable — don't feed old data to the LLM
    if is_reading_stale(sensor_data):
        return False

    # Check 1: Query specifically about coop conditions or distress?
    if is_environment_query(user_query):
        return True

    # Check 2: Resource warnings + relevant query
    resource_warning = (
        sensor_data.get("feeder_status") in ["low", "empty"] or
        sensor_data.get("waterer_status") in ["low", "empty"]
    )

    resource_keywords = ["feeder", "waterer", "refill", "empty"]
    query_lower = user_query.lower()

    if resource_warning and any(kw in query_lower for kw in resource_keywords):
        return True

    # Check 3: Critical alerts — still include sensors, but answer_query
    # will NOT use emergency mode since the query isn't about environment
    critical_params = ["temperature_status", "humidity_status", "heat_stress_index"]
    has_critical = any(
        sensor_data.get(param) == "critical"
        for param in critical_params
    )

    if has_critical:
        return True

    # Default: Skip sensors for general questions
    return False


def get_sensor_context(sensor_data: Dict) -> str:
    """
    Build sensor context string for the LLM prompt.
    Reports non-normal readings; falls back to "All readings normal."

    NOTE: The sensor team determines what is "normal", "warning", or "critical".
    We just use their labels - we don't define thresholds.
    
    Args:
        sensor_data: Current sensor readings from database
    
    Returns:
        Formatted string with relevant sensor info
    """
    
    if not sensor_data:
        return ""
    
    alerts = []
    
    # Temperature (sensor team labels this as normal/warning/critical)
    temp_status = sensor_data.get("temperature_status", "normal")
    if temp_status != "normal":
        temp_c = sensor_data.get("temperature_c")
        # Less scary wording
        status_text = "High" if temp_status == "critical" else "Elevated"
        alerts.append(
            f"🌡️ Temperature: {temp_c:.2f}°C ({status_text})"
        )
    
    # Humidity (sensor team labels this)
    humidity_status = sensor_data.get("humidity_status", "normal")
    if humidity_status != "normal":
        humidity_pct = sensor_data.get("humidity_pct")
        status_text = "High" if humidity_status == "critical" else "Elevated"
        alerts.append(
            f"💧 Humidity: {humidity_pct:.2f}% ({status_text})"
        )
    
    # Heat stress (composite indicator from sensor team)
    heat_stress = sensor_data.get("heat_stress_index", "normal")
    if heat_stress != "normal":
        # More informative, less alarming
        stress_text = "Heat stress detected" if heat_stress == "critical" else "Heat stress possible"
        alerts.append(f"🔥 {stress_text}")
    
    # Feeder
    feeder_status = sensor_data.get("feeder_status", "full")
    if feeder_status in ["low", "empty"]:
        alerts.append(f"🍗 Feeder: {feeder_status.capitalize()}")
    
    # Waterer
    waterer_status = sensor_data.get("waterer_status", "full")
    if waterer_status in ["low", "empty"]:
        alerts.append(f"💧 Waterer: {waterer_status.capitalize()}")
    
    context = "Current coop conditions:\n"
    if alerts:
        context += "\n".join(alerts)
    else:
        context += "All readings normal."
    return context


def get_critical_alerts(sensor_data: Dict) -> list:
    """
    Get list of critical alerts that require immediate attention.
    Used for automation triggers and notifications.
    
    NOTE: "Critical" status comes from sensor team's assessment.
    We format it in a calm, informative way.
    
    Args:
        sensor_data: Current sensor readings
    
    Returns:
        List of critical alert messages (calm tone)
    """
    
    if not sensor_data:
        return []
    
    critical = []
    
    # Check each parameter (using sensor team's labels)
    if sensor_data.get("temperature_status") == "critical":
        critical.append(
            f"High temperature detected: {sensor_data.get('temperature_c', 0):.2f}°C"
        )
    
    if sensor_data.get("humidity_status") == "critical":
        critical.append(
            f"High humidity detected: {sensor_data.get('humidity_pct', 0):.2f}%"
        )
    
    if sensor_data.get("heat_stress_index") == "critical":
        critical.append("Heat stress conditions present")
    
    if sensor_data.get("feeder_status") == "empty":
        critical.append("Feeder is empty")
    
    if sensor_data.get("waterer_status") == "empty":
        critical.append("Waterer is empty")
    
    return critical


# Example usage and testing
if __name__ == "__main__":
    
    # Test Case 1: Normal conditions
    print("="*60)
    print("TEST 1: Normal conditions, general question")
    print("="*60)
    
    normal_data = {
        "temperature_c": 22.3,
        "temperature_status": "normal",
        "humidity_pct": 55,
        "humidity_status": "normal",
        "heat_stress_index": "normal",
        "feeder_status": "full",
        "waterer_status": "full"
    }
    
    query1 = "How often do chickens lay eggs?"
    include1 = should_include_sensors(query1, normal_data)
    
    print(f"Query: {query1}")
    print(f"Include sensors? {include1}")
    if include1:
        print(f"Context:\n{get_sensor_context(normal_data)}")
    print()
    
    # Test Case 2: Critical temperature, environment question
    print("="*60)
    print("TEST 2: Critical temperature, environment question")
    print("="*60)
    
    critical_data = {
        "temperature_c": 35.2,
        "temperature_status": "critical",
        "humidity_pct": 85,
        "humidity_status": "critical",
        "heat_stress_index": "critical",
        "feeder_status": "low",
        "waterer_status": "empty"
    }
    
    query2 = "My chickens are panting heavily"
    include2 = should_include_sensors(query2, critical_data)
    
    print(f"Query: {query2}")
    print(f"Include sensors? {include2}")
    if include2:
        print(f"Context:\n{get_sensor_context(critical_data)}")
    print(f"\nCritical alerts: {get_critical_alerts(critical_data)}")
    print()
    
    # Test Case 3: Normal conditions, but environment question
    print("="*60)
    print("TEST 3: Normal conditions, but environment question")
    print("="*60)
    
    query3 = "What temperature is too hot for chickens?"
    include3 = should_include_sensors(query3, normal_data)
    
    print(f"Query: {query3}")
    print(f"Include sensors? {include3}")
    if include3:
        print(f"Context:\n{get_sensor_context(normal_data)}")
    print()
    
    # Test Case 4: Critical conditions, but general question
    print("="*60)
    print("TEST 4: Critical conditions, general question")
    print("="*60)
    
    query4 = "What breed should I get?"
    include4 = should_include_sensors(query4, critical_data)
    
    print(f"Query: {query4}")
    print(f"Include sensors? {include4}")
    print("(Still includes because critical alerts always show)")
    if include4:
        print(f"Context:\n{get_sensor_context(critical_data)}")
