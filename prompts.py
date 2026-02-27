"""
prompts.py — ChickenCare AI system prompts

DESIGN PHILOSOPHY
─────────────────
Be a helpful friend, not a bureaucrat. Hobby chicken keepers need
practical, concise advice — not rigid templates or constant vet referrals.

Three prompt variants exist to avoid injecting unnecessary tokens:
    SIMPLE_PROMPT    — general/factual questions, no sensor data
    MAIN_PROMPT      — question + relevant sensor readings
    EMERGENCY_PROMPT — critical sensor alert + environment/health question

Safety constraints are embedded in the persona (_SAFETY_RULES) and kept
minimal — the model should feel free to give a short, direct answer.
"""


# =============================================================================
# SHARED SAFETY RULES (persona + guardrails)
# =============================================================================
# Kept short so the model internalises the persona rather than treating
# it as a checklist to tick off in every response.

_SAFETY_RULES = """You are a friendly, knowledgeable assistant for hobby chicken keepers.
Keep answers practical and clear. Use plain language, no jargon.
Base your answer on the provided knowledge. If it doesn't fully cover the question, share the most useful practical advice you can — only admit uncertainty if you have no relevant knowledge at all.
Never suggest medications, dosages, or chemical treatments.
Always use metric units (°C, kg, cm, litres) — convert any imperial values from the knowledge base before answering.
Answer directly. Do not reproduce source labels, XML tags, or knowledge base structure in your answer."""


# =============================================================================
# SIMPLE PROMPT — general or factual questions, no sensor data
# =============================================================================
# Use when: should_include_sensors() = False
#
# No forced sections — let the model answer naturally and concisely.
# Only bring up a vet when the question genuinely involves illness/injury.

SIMPLE_PROMPT = """{safety_rules}

---
{context}
---

Question: {query}

Answer helpfully and concisely based on the knowledge above.
If this involves a sick or injured chicken, mention when a vet or experienced keeper should be contacted.
Keep it short — a few sentences or a brief list is usually enough."""


# =============================================================================
# MAIN PROMPT — question with relevant sensor context (non-critical)
# =============================================================================
# Use when: should_include_sensors() = True, has_critical = False
#
# Sensor readings are already filtered to show only non-normal values,
# so the block is compact. Integrate them naturally into the answer.

MAIN_PROMPT = """{safety_rules}

---
{context}
---

{sensor_block}

Question: {query}

Answer based on the knowledge above and the current readings.
Be direct and practical. Only mention a vet if there is a real health concern.
Keep it short."""


# =============================================================================
# EMERGENCY PROMPT — critical sensor readings + environment/health question
# =============================================================================
# Use when: has_critical = True AND sensor_context is not empty
#
# Action-first, calm tone. The keeper is likely stressed — do not amplify that.
# Knowledge base still included for context (e.g. heat stress treatment steps).

EMERGENCY_PROMPT = """{safety_rules}

The coop sensors are showing a problem right now:

{sensor_block}

---
{context}
---

Question: {query}

Give 2-3 specific actions the keeper can take right now.
Stay calm and practical. Mention a vet only if the chicken's health is genuinely at risk."""


# =============================================================================
# get_prompt() — selector used by rag_functions.py
# =============================================================================

def get_prompt(
    query: str,
    context: str,
    sensor_context: str = None,
    has_critical: bool = False,
    history: list = None,   # reserved for future use; not yet used
) -> str:
    """
    Select and format the appropriate prompt for the current situation.

    Args:
        query:          User's question
        context:        Concatenated RAG chunk text from vector/BM25 retriever
        sensor_context: Formatted sensor string from sensor_filter.py, or None
        has_critical:   True when critical alerts exist AND query is env/health-related

    Decision tree:
        has_critical AND sensor_context  →  EMERGENCY_PROMPT
        sensor_context only              →  MAIN_PROMPT
        no sensor_context                →  SIMPLE_PROMPT
    """

    if has_critical and sensor_context:
        return EMERGENCY_PROMPT.format(
            safety_rules=_SAFETY_RULES,
            sensor_block=sensor_context,
            context=context,
            query=query,
        )

    if sensor_context:
        return MAIN_PROMPT.format(
            safety_rules=_SAFETY_RULES,
            sensor_block=sensor_context,
            context=context,
            query=query,
        )

    return SIMPLE_PROMPT.format(
        safety_rules=_SAFETY_RULES,
        context=context,
        query=query,
    )
