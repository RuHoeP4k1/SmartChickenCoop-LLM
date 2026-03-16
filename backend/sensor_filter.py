"""
sensor_filter.py — Smart sensor context filtering

Decides when to include sensor data in the LLM prompt.
Core principle: only inject sensors when the user is asking about
their *current* coop situation — not for general knowledge questions.
"""

from datetime import datetime, timedelta, timezone
from typing import Dict, Optional

# If the latest reading is older than this, treat it as stale and ignore it.
STALE_THRESHOLD = timedelta(minutes=30)


# =============================================================================
# Keyword sets
# =============================================================================

# Signals that the user is asking about their coop RIGHT NOW
_CURRENT_SITUATION_KEYWORDS = [
    "my coop", "my chickens", "my flock", "my birds", "my hens",
    "right now", "at the moment", "currently", "in there",
    "is it too", "are they okay", "should i be worried",
    "panting", "lethargic", "not moving", "breathing heavy", "wings spread",
    "what are the readings", "check the coop", "how is the coop",
    "how are my", "is my",
    # Mortality / distress
    "found dead", "dead chicken", "dead hen", "dying",
    # Compound negation patterns
    "not eating", "stopped eating", "won't eat",
    "won't drink", "stopped drinking", "not drinking",
    # General concern
    "acting weird", "acting strange", "something wrong", "what's wrong with",
]

# Resource-related keywords — used in combination with low/empty status
_RESOURCE_KEYWORDS = ["feeder", "waterer", "food", "water", "refill", "empty", "refilling"]

# Broad chicken topic keywords — used only when critical conditions exist
_CHICKEN_TOPIC_KEYWORDS = [
    "chicken", "hen", "flock", "bird", "egg", "coop", "roost",
    "sick", "ill", "health", "behavior", "feed", "water",
]

# Chicken count / flock presence queries
_FLOCK_COUNT_KEYWORDS = [
    "how many chickens", "chickens inside", "inside the coop",
    "how many are inside", "flock count", "head count",
]

# Egg collection queries
_EGG_KEYWORDS = [
    "how many eggs", "egg count", "eggs today", "eggs laid",
    "collected eggs", "egg production",
]

# Door / ventilation status queries
_COOP_STATUS_KEYWORDS = [
    "is the door", "door open", "door closed", "is it open",
    "ventilation", "fan on", "is it ventilated", "air in the coop",
]

# Behavioral / symptom keywords — these almost always warrant sensor context
_BEHAVIOR_KEYWORDS = [
    "molting", "moulting", "losing feathers", "feather loss", "bald spots",
    "broody", "sitting on eggs", "won't leave nest",
    "aggressive", "bullying", "pecking each other", "fighting",
    "limping", "can't walk", "not walking",
    "ruffled feathers", "puffed up", "fluffed up",
    "diarrhea", "runny droppings", "watery droppings",
    "coughing", "sneezing", "wheezing", "gasping", "rattling",
    "listless", "not active", "huddling",
    "pale comb", "blue comb", "swollen eyes", "discharge",
    "worms", "mites", "lice", "parasites",
]

# Signals that a query is encyclopedic / general knowledge (suppress behavior trigger)
_GENERAL_KNOWLEDGE_SIGNALS = [
    "what is ", "what are ", "what causes ", "how to ", "how do ",
    "why do chickens", "why does a chicken", "tell me about",
    "explain ", "define ", "in general",
]


# =============================================================================
# Staleness check
# =============================================================================

def is_reading_stale(sensor_data: Dict) -> bool:
    """
    Return True if the reading's timestamp is older than STALE_THRESHOLD,
    or if there is no timestamp.
    """
    ts = sensor_data.get("timestamp")
    if ts is None:
        return True

    if isinstance(ts, str):
        ts = datetime.fromisoformat(ts)

    now = datetime.now(ts.tzinfo)  # match tz-awareness of the DB value
    return (now - ts) > STALE_THRESHOLD


# =============================================================================
# Environment / current-coop query detection
# =============================================================================

def is_environment_query(user_query: str) -> bool:
    """
    Return True if the query is specifically about the user's current coop
    conditions or visible distress symptoms — i.e. sensor data is relevant.

    This is narrower than before: "what temperature is too hot for chickens?"
    is a general knowledge question and returns False.
    "Is my coop too hot right now?" returns True.
    """
    query_lower = user_query.lower()
    return any(kw in query_lower for kw in _CURRENT_SITUATION_KEYWORDS)


# =============================================================================
# Main filtering decision
# =============================================================================

def should_include_sensors(user_query: str, sensor_data: Dict) -> bool:
    """
    Decide whether to include sensor data in the LLM prompt.

    Rules (in priority order):
    1. Stale / missing data → never include
    2. User is asking about their current coop situation → include
    3. Resource question (feeder/water) AND resources are low/empty → include
    4. Critical conditions AND question is about chickens/coop → include
    5. General knowledge question → skip sensors
    """
    if not sensor_data:
        return False

    if is_reading_stale(sensor_data):
        return False

    query_lower = user_query.lower()

    # Rule 2: User is asking about their coop right now
    if any(kw in query_lower for kw in _CURRENT_SITUATION_KEYWORDS):
        return True

    # Rule 2.5: Behavioral concern — symptoms that correlate with environment
    # (but not general knowledge questions like "what causes molting?")
    if any(kw in query_lower for kw in _BEHAVIOR_KEYWORDS):
        if not any(sig in query_lower for sig in _GENERAL_KNOWLEDGE_SIGNALS):
            return True

    # Rule 3: Resource question + resource is actually low or empty
    resource_low = (
        sensor_data.get("feeder_status") in ["low", "empty"] or
        sensor_data.get("waterer_status") in ["low", "empty"]
    )
    if resource_low and any(kw in query_lower for kw in _RESOURCE_KEYWORDS):
        return True

    # Rule 4: Critical conditions, but ONLY if the question is about chickens/coop
    # (avoids injecting sensors for "what breed should I get?" when coop is critical)
    critical_params = [
        "temperature_status", "humidity_status", "heat_stress_index",
        "h2s_level", "mold_risk_status",
    ]
    has_critical = any(sensor_data.get(p) == "critical" for p in critical_params)

    if has_critical and any(kw in query_lower for kw in _CHICKEN_TOPIC_KEYWORDS):
        return True

    # Rule 5: Flock count / egg / door / ventilation question → include (we have that data)
    if any(kw in query_lower for kw in _FLOCK_COUNT_KEYWORDS):
        return True
    if any(kw in query_lower for kw in _EGG_KEYWORDS):
        return True
    if any(kw in query_lower for kw in _COOP_STATUS_KEYWORDS):
        return True

    return False


# =============================================================================
# Sensor context formatting
# =============================================================================

def get_sensor_context(sensor_data: Dict) -> str:
    """
    Build a compact sensor context string for the LLM prompt.

    Only reports readings that are non-normal / non-full. Does NOT dump
    all sensor values — chickens_inside, egg_count, door, ventilation are
    only included when notable (door open = notable; ventilation always on = not).

    Returns:
        Formatted string with current readings, or "All coop readings normal."
    """
    if not sensor_data:
        return ""

    alerts = []

    # Temperature
    temp_status = sensor_data.get("temperature_status", "normal")
    if temp_status != "normal":
        temp_c = sensor_data.get("temperature_c")
        if temp_c is not None:
            alerts.append(f"Temperature: {temp_c:.1f}°C ({temp_status})")

    # Humidity
    humidity_status = sensor_data.get("humidity_status", "normal")
    if humidity_status != "normal":
        humidity_pct = sensor_data.get("humidity_pct")
        if humidity_pct is not None:
            alerts.append(f"Humidity: {humidity_pct:.0f}% ({humidity_status})")

    # Heat stress (composite)
    heat_stress = sensor_data.get("heat_stress_index", "normal")
    if heat_stress != "normal":
        alerts.append(f"Heat stress: {heat_stress}")

    # Feeder — include percentage when available
    feeder_status = sensor_data.get("feeder_status", "full")
    if feeder_status in ["low", "empty"]:
        pct = sensor_data.get("feeder_pct")
        pct_str = f" ({pct:.0f}%)" if pct is not None else ""
        alerts.append(f"Feeder: {feeder_status}{pct_str}")

    # Waterer
    waterer_status = sensor_data.get("waterer_status", "full")
    if waterer_status in ["low", "empty"]:
        pct = sensor_data.get("waterer_pct")
        pct_str = f" ({pct:.0f}%)" if pct is not None else ""
        alerts.append(f"Waterer: {waterer_status}{pct_str}")

    # H2S gas
    h2s_level = sensor_data.get("h2s_level", "normal")
    if h2s_level != "normal":
        h2s_ppm = sensor_data.get("h2s_ppm")
        ppm_str = f" ({h2s_ppm:.0f} ppm)" if h2s_ppm is not None else ""
        alerts.append(f"H2S gas: {h2s_level}{ppm_str}")

    # Mold risk
    mold_risk = sensor_data.get("mold_risk_status", "normal")
    if mold_risk != "normal":
        alerts.append(f"Mold risk: {mold_risk}")

    # Door — only notable when open
    if sensor_data.get("door_open"):
        alerts.append("Coop door: open")

    # Chickens inside — always include (useful operational context)
    chickens = sensor_data.get("chickens_inside")
    if chickens is not None:
        alerts.append(f"Chickens inside coop: {chickens}")

    # Egg count — include when eggs have been laid
    eggs = sensor_data.get("egg_count")
    if eggs is not None and eggs > 0:
        alerts.append(f"Eggs detected: {eggs}")

    # Crowding assessment — always include when available
    crowding = sensor_data.get("crowding_assessment")
    if crowding is not None:
        alerts.append(f"Crowding assessment: {crowding}")

    # Ventilation — include when on (relevant for temp/H2S alerts)
    if sensor_data.get("ventilation_on"):
        alerts.append("Ventilation: on")

    if alerts:
        return "Current coop readings:\n" + "\n".join(f"- {a}" for a in alerts)
    return "All coop readings normal."


# =============================================================================
# Critical alert list (used by scheduler and answer_query)
# =============================================================================

def get_critical_alerts(sensor_data: Dict) -> list:
    """
    Return a list of critical alert messages (calm tone).
    Used by the scheduler for automation triggers and by answer_query
    to decide whether to use the emergency prompt.
    """
    if not sensor_data:
        return []

    critical = []

    if sensor_data.get("temperature_status") == "critical":
        temp = sensor_data.get("temperature_c", 0)
        critical.append(f"High temperature: {temp:.1f}°C")

    if sensor_data.get("humidity_status") == "critical":
        hum = sensor_data.get("humidity_pct", 0)
        critical.append(f"High humidity: {hum:.0f}%")

    if sensor_data.get("heat_stress_index") == "critical":
        critical.append("Heat stress conditions present")

    if sensor_data.get("feeder_status") == "empty":
        critical.append("Feeder is empty")

    if sensor_data.get("waterer_status") == "empty":
        critical.append("Waterer is empty")

    if sensor_data.get("h2s_level") == "critical":
        ppm = sensor_data.get("h2s_ppm")
        ppm_str = f": {ppm:.0f} ppm" if ppm is not None else ""
        critical.append(f"Dangerous H2S gas detected{ppm_str}")

    if sensor_data.get("mold_risk_status") == "critical":
        critical.append("Critical mold risk conditions")

    return critical


# =============================================================================
# Manual test
# =============================================================================

if __name__ == "__main__":
    _now = datetime.now()

    normal = {
        "timestamp": _now,
        "temperature_c": 22.3, "temperature_status": "normal",
        "humidity_pct": 55, "humidity_status": "normal",
        "heat_stress_index": "normal", "feeder_status": "full",
        "waterer_status": "full", "feeder_pct": 80, "waterer_pct": 75,
        "h2s_level": "normal", "mold_risk_status": "normal",
        "door_open": False,
    }
    critical = {**normal,
        "temperature_c": 35.2, "temperature_status": "critical",
        "humidity_pct": 85, "humidity_status": "critical",
        "heat_stress_index": "critical",
        "feeder_status": "empty", "feeder_pct": 2,
        "waterer_status": "empty", "waterer_pct": 3,
    }

    tests = [
        ("How often do chickens lay eggs?", normal, False),
        ("What temperature is too hot for chickens?", normal, False),
        ("Is my coop too hot right now?", normal, True),
        ("My chickens are panting", normal, True),
        ("What breed should I get?", critical, False),
        ("Are my chickens okay?", critical, True),
        # Broader intent detection
        ("my chickens won't drink", normal, True),
        ("are they molting?", normal, True),
        ("one chicken is coughing", normal, True),
        ("what causes molting in chickens?", normal, False),
        ("my chickens have diarrhea, what causes this?", normal, True),
        ("how to treat mites on chickens", normal, False),
    ]

    for query, data, expected in tests:
        result = should_include_sensors(query, data)
        status = "OK" if result == expected else "FAIL"
        print(f"[{status}] '{query}' → include={result} (expected={expected})")
        if result:
            print(get_sensor_context(data))
        print()
