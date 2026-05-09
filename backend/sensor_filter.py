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
# Risk level helpers (risk_snapshots text levels)
# =============================================================================
# heat_risk_level format from risk_calculation.py:
#   "LOW - Monitor" | "MEDIUM - Elevated risk - Prepare to act" | "HIGH - Take action now"
# mold_risk_level format from risk_calculation.py:
#   "low" | "medium" | "high" | "severe"

def heat_is_critical(level: Optional[str]) -> bool:
    return bool(level) and level.upper().startswith("HIGH")


def heat_is_non_normal(level: Optional[str]) -> bool:
    return bool(level) and not level.upper().startswith("LOW")


def mold_is_critical(level: Optional[str]) -> bool:
    return (level or "").lower() in ("high", "severe")


def mold_is_non_normal(level: Optional[str]) -> bool:
    v = (level or "").lower()
    return bool(v) and v != "low"


# Back-compat aliases (internal callers may still use the underscore names)
_heat_is_critical = heat_is_critical
_heat_is_non_normal = heat_is_non_normal
_mold_is_critical = mold_is_critical
_mold_is_non_normal = mold_is_non_normal


# =============================================================================
# Keyword sets
# =============================================================================

# Signals that the user is asking about their coop RIGHT NOW
_CURRENT_SITUATION_KEYWORDS = [
    # Ownership — singular and plural
    "my coop", "my chickens", "my flock", "my birds", "my hens",
    "my chicken", "my hen", "my bird", "my rooster",
    # Time anchors
    "right now", "at the moment", "currently", "in there",
    "last night", "overnight", "this morning", "this afternoon", "tonight", "today",
    # Direct concern phrasing
    "is it too", "are they okay", "should i be worried",
    "are they safe", "is it safe", "worried about",
    "safe for them", "dangerous for them", "okay for them",
    # Observed distress symptoms
    "panting", "lethargic", "not moving", "breathing heavy", "wings spread",
    # Operational queries
    "what are the readings", "check the coop", "how is the coop",
    "how are my", "is my", "how's the coop", "how's my coop",
    "check on", "status of", "anything wrong", "any problems", "any issues",
    "what's happening", "how are things", "is everything okay", "is everything alright",
    # Natural environment phrasing
    "how hot", "how cold", "how warm", "how humid",
    "what's the temperature", "what is the temperature",
    "what's the humidity", "what is the humidity",
    "temperature in the coop", "humidity in the coop",
    "is the coop warm", "is the coop cold", "is the coop hot",
    "too hot in", "too cold in", "too humid",
    # Air quality / smell — descriptive phrases that are clearly situational
    # (bare "ammonia" alone is NOT here — "what causes ammonia" is general knowledge)
    "smell bad", "stinks", "it smells", "smells bad", "smells like",
    "stuffy in", "muggy in", "damp in", "condensation in", "foggy in",
    # Mortality / distress
    "found dead", "dead chicken", "dead hen", "dying",
    # Compound negation patterns
    "not eating", "stopped eating", "won't eat",
    "won't drink", "stopped drinking", "not drinking",
    # Resource status questions — always show sensor reading (even if normal)
    "run out of", "run out of food", "run out of water", "did they run out",
    # General concern
    "acting weird", "acting strange", "something wrong", "what's wrong with",
]

# Resource-related keywords — used in combination with low/empty status
_RESOURCE_KEYWORDS = [
    "feeder", "waterer", "food", "water", "refill", "empty", "refilling",
    "hungry", "thirsty", "run out", "running low", "running out",
    "out of food", "out of water", "need more food", "need more water",
    "topped up", "filled up",
]

# Broad chicken topic keywords — used only when critical conditions exist
_CHICKEN_TOPIC_KEYWORDS = [
    "chicken", "hen", "flock", "bird", "egg", "coop", "roost",
    "sick", "ill", "health", "behavior", "feed", "water",
    "disease", "illness", "infection", "outbreak",
]

# Chicken count / flock presence queries
_FLOCK_COUNT_KEYWORDS = [
    "how many chickens", "chickens inside", "inside the coop",
    "how many are inside", "flock count", "head count",
]

# Egg collection queries — narrowly scoped to the user's own coop/flock
# "how many eggs" alone is too broad (matches "how many eggs do chickens lay?" = general knowledge)
_EGG_KEYWORDS = [
    "egg count", "eggs today", "eggs laid", "collected eggs", "egg production",
    "how many eggs did", "how many eggs are", "any eggs today", "any eggs yet",
]

# Door / ventilation status queries
_COOP_STATUS_KEYWORDS = [
    "is the door", "door open", "door closed", "is it open",
    "ventilation", "fan on", "is it ventilated", "air in the coop",
    # Forgot-to-close patterns
    "did i close", "did i leave", "left the door", "close the door",
    "shut the coop", "lock the coop",
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

# Air quality keywords that are ambiguous — "ammonia" alone appears in both
# situational ("there's ammonia in my coop") and encyclopedic ("what causes ammonia")
# questions. Checked with general knowledge signal gating (like _BEHAVIOR_KEYWORDS).
_AIR_QUALITY_KEYWORDS = [
    "ammonia", "smells", "stuffy", "muggy", "condensation", "foggy",
    "h2s", "hydrogen sulfide", "gas in", "gas levels",
]

# Signals that a query is encyclopedic / general knowledge (suppress behavior trigger)
_GENERAL_KNOWLEDGE_SIGNALS = [
    "what is ", "what are ", "what causes ", "how to ", "how do ",
    "why do chickens", "why does a chicken", "tell me about",
    "explain ", "define ", "in general",
    # Breed / acquisition questions — never need live sensor data
    "what breed", "which breed", "best breed", "can chickens",
    "do chickens ", "should chickens",
    # Age / lifecycle questions
    "at what age", "how old", "when do chickens",
    # Generic advice framing
    "is it normal", "is it normal for chickens",
    "in general how", "generally how",
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

    query_lower = user_query.lower()

    # Rule 2: User is asking about their coop right now
    if any(kw in query_lower for kw in _CURRENT_SITUATION_KEYWORDS):
        return True

    # Rule 2.5: Behavioral concern — symptoms that correlate with environment
    # (but not general knowledge questions like "what causes molting?")
    if any(kw in query_lower for kw in _BEHAVIOR_KEYWORDS):
        if not any(sig in query_lower for sig in _GENERAL_KNOWLEDGE_SIGNALS):
            return True

    # Rule 2.6: Air quality / smell — ambiguous keywords gated by knowledge signals
    # "there's ammonia in my coop" → include; "what causes ammonia?" → exclude
    if any(kw in query_lower for kw in _AIR_QUALITY_KEYWORDS):
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
    critical_params = ["temperature_status", "humidity_status", "h2s_level"]
    has_critical = (
        any(sensor_data.get(p) == "critical" for p in critical_params)
        or _heat_is_critical(sensor_data.get("heat_risk_level"))
        or _mold_is_critical(sensor_data.get("mold_risk_level"))
    )

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
# Query → sensor group relevance
# =============================================================================

_ALL_GROUPS = {'climate', 'air_quality', 'resources', 'flock', 'infrastructure'}

# Broad queries that warrant all sensor context
_BROAD_QUERY_SIGNALS = [
    "how's the coop", "how is the coop", "how are things",
    "check on", "check the coop", "status of", "anything wrong",
    "any problems", "any issues", "what's happening", "what are the readings",
    "is everything okay", "is everything alright",
    "how are my", "are they okay", "are they safe",
    "should i be worried", "worried about",
]

# Topic keywords → which sensor groups are relevant to the query.
# Substring matching; a query can match multiple groups.
_QUERY_SENSOR_RELEVANCE = {
    'climate': [
        'hot', 'cold', 'warm', 'cool', 'temperature', 'heat',
        'humid', 'dry', 'weather', 'freez', 'frost',
        # Heat-stress symptoms
        'panting', 'lethargic', 'lethargy', 'wings spread', 'huddl', 'shiver',
        # General health → include environmental context
        'sick', 'ill', 'health', 'disease',
    ],
    'air_quality': [
        'smell', 'stink', 'odor', 'odour', 'ammonia', 'air',
        'stuffy', 'breath', 'cough', 'sneez', 'gasp', 'wheez', 'rattling',
        'mold', 'mould', 'damp', 'condensat', 'foggy', 'muggy',
        'gas', 'co2', 'carbon dioxide', 'nh3', 'h2s', 'hydrogen sulfide',
        'ventilat', 'fan',
        # General health → include air quality
        'sick', 'ill', 'health', 'disease',
    ],
    'resources': [
        'food', 'feed', 'water', 'drink', 'hungry', 'thirsty',
        'eat', 'refill', 'empty', 'feeder', 'waterer',
        'topped up', 'filled up', 'run out', 'running low',
    ],
    'flock': [
        'chicken', 'hen', 'bird', 'flock', 'rooster',
        'crowd', 'space', 'peck', 'fight', 'aggressiv', 'bully',
        'egg', 'lay', 'laid', 'inside', 'count', 'how many',
        'broody', 'molt', 'moult', 'feather',
        'sick', 'ill', 'health', 'disease', 'infect',
        'limp', 'walk', 'diarrhea', 'dropping', 'worm', 'mite', 'lice', 'parasit',
        'pale comb', 'blue comb', 'swollen', 'discharge', 'dead', 'dying',
    ],
    'infrastructure': [
        'door', 'close', 'open', 'lock', 'shut',
        'ventilat', 'fan',
        'left the', 'did i',
    ],
}


def _relevant_groups(query: Optional[str]) -> set:
    """Determine which sensor groups are relevant to the user's query."""
    if not query:
        return _ALL_GROUPS

    q = query.lower()

    # Broad concern queries → all groups
    if any(sig in q for sig in _BROAD_QUERY_SIGNALS):
        return _ALL_GROUPS

    groups = set()
    for group, keywords in _QUERY_SENSOR_RELEVANCE.items():
        if any(kw in q for kw in keywords):
            groups.add(group)

    # No specific match → all groups (safe fallback)
    return groups or _ALL_GROUPS


# =============================================================================
# Sensor context formatting
# =============================================================================

def get_sensor_context(sensor_data: Dict, query: Optional[str] = None) -> str:
    """
    Build a compact sensor context string for the LLM prompt.

    When a query is provided, only includes sensor groups relevant to the
    query topic (climate, air_quality, resources, flock, infrastructure).
    When no query is given, includes all non-normal readings (full dump).

    Returns:
        Formatted string with current readings, or "All coop readings normal."
    """
    if not sensor_data:
        return ""

    groups = _relevant_groups(query)
    alerts = []

    # Temperature — climate
    if 'climate' in groups:
        temp_status = sensor_data.get("temperature_status", "normal")
        if temp_status != "normal":
            temp_c = sensor_data.get("temperature_c")
            if temp_c is not None:
                alerts.append(f"Temperature: {temp_c:.1f}°C ({temp_status})")

    # Humidity — climate
    if 'climate' in groups:
        humidity_status = sensor_data.get("humidity_status", "normal")
        if humidity_status != "normal":
            humidity_pct = sensor_data.get("humidity_pct")
            if humidity_pct is not None:
                alerts.append(f"Humidity: {humidity_pct:.0f}% ({humidity_status})")

    # Heat risk + THI (from risk_snapshots) — climate
    if 'climate' in groups:
        heat_level = sensor_data.get("heat_risk_level") or ""
        thi = sensor_data.get("thi_current")
        if _heat_is_non_normal(heat_level):
            level_short = heat_level.split(" - ")[0] if " - " in heat_level else heat_level
            if thi is not None:
                alerts.append(f"Heat risk: {level_short} (THI {float(thi):.1f})")
            else:
                alerts.append(f"Heat risk: {level_short}")
        elif thi is not None:
            alerts.append(f"THI: {float(thi):.1f} (normal)")

    # Feeder — resources
    if 'resources' in groups:
        feeder_status = sensor_data.get("feeder_status", "full")
        if feeder_status in ["low", "empty"]:
            pct = sensor_data.get("feeder_pct")
            pct_str = f" ({pct:.0f}%)" if pct is not None else ""
            alerts.append(f"Feeder: {feeder_status}{pct_str}")

    # Waterer — resources
    if 'resources' in groups:
        waterer_status = sensor_data.get("waterer_status", "full")
        if waterer_status in ["low", "empty"]:
            pct = sensor_data.get("waterer_pct")
            pct_str = f" ({pct:.0f}%)" if pct is not None else ""
            alerts.append(f"Waterer: {waterer_status}{pct_str}")

    # H2S gas — air_quality
    if 'air_quality' in groups:
        h2s_level = sensor_data.get("h2s_level", "normal")
        if h2s_level != "normal":
            h2s_ppm = sensor_data.get("h2s_ppm")
            ppm_str = f" ({h2s_ppm:.0f} ppm)" if h2s_ppm is not None else ""
            alerts.append(f"H2S gas: {h2s_level}{ppm_str}")

    # CO2 — air_quality
    if 'air_quality' in groups:
        co2_level = sensor_data.get("co2_level", "normal")
        if co2_level != "normal":
            co2_ppm = sensor_data.get("co2_ppm")
            ppm_str = f" ({co2_ppm:.0f} ppm)" if co2_ppm is not None else ""
            alerts.append(f"CO2: {co2_level}{ppm_str}")

    # NH3 (ammonia) — air_quality
    if 'air_quality' in groups:
        nh3_level = sensor_data.get("nh3_level", "normal")
        if nh3_level != "normal":
            nh3_ppm = sensor_data.get("nh3_ppm")
            ppm_str = f" ({nh3_ppm:.1f} ppm)" if nh3_ppm is not None else ""
            alerts.append(f"NH3 (ammonia): {nh3_level}{ppm_str}")

    # Mold risk (from risk_snapshots) — air_quality
    if 'air_quality' in groups:
        mold_level = sensor_data.get("mold_risk_level")
        if _mold_is_non_normal(mold_level):
            alerts.append(f"Mold risk: {mold_level}")

    # Crowding verdict (from crowding table) — flock
    if 'flock' in groups:
        crowding_verdict = sensor_data.get("crowding_verdict")
        if crowding_verdict and not crowding_verdict.upper().startswith("NOT OVERCROWDED"):
            alerts.append(f"Crowding: {crowding_verdict}")

    # Door — infrastructure (only notable when open)
    if 'infrastructure' in groups:
        if sensor_data.get("door_open"):
            alerts.append("Coop door: open")

    # Chickens inside — flock
    if 'flock' in groups:
        chickens = sensor_data.get("number_of_chickens") or sensor_data.get("chickens_inside")
        if chickens is not None:
            alerts.append(f"Chickens inside coop: {chickens}")

    # Egg count — flock
    if 'flock' in groups:
        eggs = sensor_data.get("egg_count")
        if eggs is not None and eggs > 0:
            alerts.append(f"Eggs detected: {eggs}")

    # Ventilation — infrastructure
    if 'infrastructure' in groups:
        if sensor_data.get("ventilation_on"):
            alerts.append("Ventilation: on")

    ts = sensor_data.get("timestamp")
    if is_reading_stale(sensor_data) and ts is not None:
        header = f"Last sensor measurement (recorded: {ts}):"
    else:
        header = "Current coop readings:"

    # contributing_factors from risk_snapshots — include when climate or air_quality relevant
    risk_factors = sensor_data.get("risk_factors") or ""
    risk_context_line = ""
    if risk_factors and ('climate' in groups or 'air_quality' in groups):
        risk_context_line = f"\nRisk context: {risk_factors}"

    if alerts:
        return header + "\n" + "\n".join(f"- {a}" for a in alerts) + risk_context_line
    return f"{header}\n- All readings normal."


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
        temp = sensor_data.get("temperature_c") or 0
        critical.append(f"High temperature: {temp:.1f}°C")

    if sensor_data.get("humidity_status") == "critical":
        hum = sensor_data.get("humidity_pct") or 0
        direction = "Low" if hum < 50 else "High"
        critical.append(f"{direction} humidity: {hum:.0f}%")

    if _heat_is_critical(sensor_data.get("heat_risk_level")):
        thi = sensor_data.get("thi_current")
        thi_str = f" (THI {float(thi):.1f})" if thi is not None else ""
        critical.append(f"High heat stress risk{thi_str}")

    if sensor_data.get("feeder_status") == "empty":
        critical.append("Feeder is empty")

    if sensor_data.get("waterer_status") == "empty":
        critical.append("Waterer is empty")

    if sensor_data.get("h2s_level") == "critical":
        ppm = sensor_data.get("h2s_ppm")
        ppm_str = f": {ppm:.0f} ppm" if ppm is not None else ""
        critical.append(f"Dangerous H2S gas detected{ppm_str}")

    if _mold_is_critical(sensor_data.get("mold_risk_level")):
        critical.append(f"Critical mold risk ({sensor_data.get('mold_risk_level')})")

    if sensor_data.get("co2_level") == "critical":
        ppm = sensor_data.get("co2_ppm")
        ppm_str = f": {ppm:.0f} ppm" if ppm is not None else ""
        critical.append(f"Dangerous CO2 level{ppm_str}")

    if sensor_data.get("nh3_level") == "critical":
        ppm = sensor_data.get("nh3_ppm")
        ppm_str = f": {ppm:.1f} ppm" if ppm is not None else ""
        critical.append(f"Dangerous NH3 (ammonia) level{ppm_str}")

    crowding_verdict = sensor_data.get("crowding_verdict") or ""
    if crowding_verdict.upper().startswith("OVERCROWDED"):
        critical.append(f"Coop overcrowded: {crowding_verdict}")

    return critical


# =============================================================================
# SENSOR SUMMARY + LLM ROUTER
# =============================================================================

import os as _os


def format_sensor_summary(sensor_data: Dict) -> str:
    """Compact one-line sensor summary for the LLM routing classifier."""
    sd = sensor_data
    parts = []
    parts.append(f"temp={sd.get('temperature_c','?')}C({sd.get('temperature_status','?')})")
    parts.append(f"humidity={sd.get('humidity_pct','?')}%({sd.get('humidity_status','?')})")
    heat_level = sd.get("heat_risk_level") or "?"
    heat_short = heat_level.split(" - ")[0] if " - " in heat_level else heat_level
    thi = sd.get("thi_current")
    parts.append(f"heat_risk={heat_short}" + (f"(THI {float(thi):.1f})" if thi is not None else ""))
    parts.append(f"feeder={sd.get('feeder_status','?')}({sd.get('feeder_pct','?')}%)")
    parts.append(f"waterer={sd.get('waterer_status','?')}({sd.get('waterer_pct','?')}%)")
    h2s_ppm = sd.get("h2s_ppm")
    h2s = sd.get("h2s_level", "?") + (f"({h2s_ppm}ppm)" if h2s_ppm else "")
    parts.append(f"h2s={h2s}")
    co2_ppm = sd.get("co2_ppm")
    parts.append(f"co2={sd.get('co2_level','?')}" + (f"({co2_ppm:.0f}ppm)" if co2_ppm else ""))
    nh3_ppm = sd.get("nh3_ppm")
    parts.append(f"nh3={sd.get('nh3_level','?')}" + (f"({nh3_ppm:.1f}ppm)" if nh3_ppm else ""))
    parts.append(f"mold_risk={sd.get('mold_risk_level','?')}")
    crowding = sd.get("crowding_verdict") or "unknown"
    parts.append(f"crowding={crowding.split(' - ')[0].split(' =')[0]}")
    parts.append(f"door={'open' if sd.get('door_open') else 'closed'}")
    parts.append(f"chickens={sd.get('number_of_chickens','N/A')}")
    parts.append(f"eggs={sd.get('egg_count', 0)}")
    return ", ".join(parts)


_LLM_ROUTER_PROMPT = """\
You decide whether live coop sensor readings should be included when answering \
a chicken-keeping question. Return ONLY the word INCLUDE or EXCLUDE. No explanation.

RULES (apply in order, stop at first match):
1. User asks about their own coop conditions right now ("my coop", "right now", \
"is my feeder okay", "did I close the door", symptoms like panting/lethargy, \
status updates) -> INCLUDE
2. User describes a health/behavior concern (molting, limping, diarrhea, sneezing) \
NOT as a general knowledge question ("what causes...", "how to...") -> INCLUDE
3. User mentions air quality/smells (ammonia, stuffy, stinks, ...) NOT encyclopedically -> INCLUDE
4. Resource (feeder/waterer) is low/empty AND user mentions food/water/feeder/waterer -> INCLUDE
5. Any sensor is CRITICAL AND question is about chickens/coop/health/eggs/disease -> INCLUDE
6. User asks about flock count, egg count, or door/ventilation status -> INCLUDE
7. General knowledge (breed advice, "what is...", lifecycle, design of a coop) with no personal coop \
reference -> EXCLUDE

{passages_block}SENSOR STATUS: {sensor_summary}
USER QUESTION: {query}
DECISION:"""

_router_llm = None
_router_model_id: Optional[str] = None


def llm_route_sensors(
    user_query: str,
    sensor_data: Dict,
    model: str = None,
    retrieved_passages: list = None,
) -> bool:
    """
    Use an LLM to decide whether sensor data should be injected into the prompt.

    Falls back to keyword routing (should_include_sensors) on parse error or exception.
    Configure via env vars:
        SENSOR_ROUTER_MODEL  — override model for classifier only (defaults to OLLAMA_MODEL)
    """
    global _router_llm, _router_model_id

    if model is None:
        model = _os.getenv("SENSOR_ROUTER_MODEL") or _os.getenv("OLLAMA_MODEL", "smollm2:1.7b")

    if _router_llm is None or _router_model_id != model:
        if model.startswith("openrouter/"):
            from langchain_openai import ChatOpenAI
            _router_llm = ChatOpenAI(
                model=model.removeprefix("openrouter/"),
                base_url="https://openrouter.ai/api/v1",
                api_key=_os.getenv("OPENROUTER_API_KEY"),
                temperature=0.1,
                max_tokens=10,
            )
        else:
            from langchain_ollama import OllamaLLM
            _router_llm = OllamaLLM(model=model, temperature=0.1, num_predict=10)
        _router_model_id = model

    try:
        passages_block = ""
        if retrieved_passages:
            snippets = [
                f"- {doc.page_content[:200].replace(chr(10), ' ')}"
                for doc in retrieved_passages[:2]
            ]
            passages_block = "RELEVANT KNOWLEDGE BASE PASSAGES:\n" + "\n".join(snippets) + "\n\n"

        prompt = _LLM_ROUTER_PROMPT.format(
            passages_block=passages_block,
            sensor_summary=format_sensor_summary(sensor_data),
            query=user_query,
        )
        response = _router_llm.invoke(prompt)
        raw = (response.content if hasattr(response, "content") else str(response)).strip().upper()
        if raw.startswith("INCL"):
            return True
        if raw.startswith("EXCL"):
            return False
        # Parse error — fall back to keyword baseline
        return should_include_sensors(user_query, sensor_data)
    except Exception:
        return should_include_sensors(user_query, sensor_data)


# =============================================================================
# Manual test
# =============================================================================

if __name__ == "__main__":
    _now = datetime.now()

    normal = {
        "timestamp": _now,
        "temperature_c": 22.3, "temperature_status": "normal",
        "humidity_pct": 55, "humidity_status": "normal",
        "heat_risk_level": "LOW - Monitor", "thi_current": 18.5,
        "feeder_status": "full",
        "waterer_status": "full", "feeder_pct": 80, "waterer_pct": 75,
        "h2s_level": "normal", "mold_risk_level": "low",
        "door_open": False,
    }
    critical = {**normal,
        "temperature_c": 35.2, "temperature_status": "critical",
        "humidity_pct": 85, "humidity_status": "critical",
        "heat_risk_level": "HIGH - Take action now", "thi_current": 31.2,
        "feeder_status": "empty", "feeder_pct": 2,
        "waterer_status": "empty", "waterer_pct": 3,
    }

    tests = [
        # Original cases
        ("How often do chickens lay eggs?", normal, False),
        ("What temperature is too hot for chickens?", normal, False),
        ("Is my coop too hot right now?", normal, True),
        ("My chickens are panting", normal, True),
        ("What breed should I get?", critical, False),
        ("Are my chickens okay?", critical, True),
        ("my chickens won't drink", normal, True),
        ("are they molting?", normal, True),
        ("one chicken is coughing", normal, True),
        ("what causes molting in chickens?", normal, False),
        ("my chickens have diarrhea, what causes this?", normal, True),
        ("how to treat mites on chickens", normal, False),
        # Singular forms
        ("my chicken seems off today", normal, True),
        ("my hen is not moving", normal, True),
        # Natural environment phrasing
        ("how hot is it in the coop?", normal, True),
        ("what's the temperature in the coop?", normal, True),
        ("is it too humid?", normal, True),
        # Operational / status queries
        ("can you check on my chickens?", normal, True),
        ("anything wrong in the coop?", normal, True),
        ("how's the coop doing?", normal, True),
        # Air quality
        ("it smells bad in the coop", normal, True),
        ("there's ammonia in there", normal, True),
        # Door / forgot-to-close
        ("did i close the coop door?", normal, True),
        ("did i leave the door open?", normal, True),
        # Resource natural language
        ("are my chickens thirsty?", normal, True),
        ("did they run out of food?", normal, True),
        # General knowledge suppression (should NOT include)
        ("what breed lays the most eggs?", normal, False),
        ("can chickens eat tomatoes?", normal, False),
        ("when do chickens start laying?", normal, False),
        ("do chickens need sunlight?", normal, False),
        # Critical + chicken topic still includes
        ("how do i prevent disease?", critical, True),
    ]

    for query, data, expected in tests:
        result = should_include_sensors(query, data)
        status = "OK" if result == expected else "FAIL"
        print(f"[{status}] '{query}' -> include={result} (expected={expected})")
        if result:
            print(get_sensor_context(data))
        print()

    # --- Selective context tests ---
    print("=" * 60)
    print("SELECTIVE CONTEXT TESTS")
    print("=" * 60)

    multi_alert = {
        **normal,
        "temperature_c": 35.2, "temperature_status": "critical",
        "humidity_pct": 85, "humidity_status": "warning",
        "heat_risk_level": "HIGH - Take action now", "thi_current": 31.2,
        "co2_ppm": 1800, "co2_level": "warning",
        "nh3_ppm": 18.5, "nh3_level": "warning",
        "feeder_status": "empty", "feeder_pct": 2,
        "crowding_verdict": "OVERCROWDED = 15 chickens in 10m²",
        "door_open": True,
        "number_of_chickens": 15,
        "egg_count": 4,
        "ventilation_on": True,
    }

    selective_tests = [
        ("how many eggs today?",         {'flock'}),
        ("is the door closed?",          {'infrastructure'}),
        ("it smells like ammonia",        {'air_quality'}),
        ("is it too hot in the coop?",    {'climate'}),
        ("did they run out of water?",    {'resources'}),
        ("how's the coop?",              _ALL_GROUPS),   # broad → all
        (None,                            _ALL_GROUPS),   # no query → all
    ]

    for query, expected_groups in selective_tests:
        groups = _relevant_groups(query)
        ctx = get_sensor_context(multi_alert, query=query)
        match = groups == expected_groups
        status = "OK" if match else "FAIL"
        print(f"\n[{status}] query={query!r}")
        print(f"  groups: {sorted(groups)}  (expected {sorted(expected_groups)})")
        print(f"  context:\n{ctx}")
