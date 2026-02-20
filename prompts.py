"""
System prompts for ChickenCare AI
Simple, realistic prompts based on actual chicken-keeping knowledge
"""


# Main system prompt - used for queries where sensor data is relevant
SYSTEM_PROMPT = """You are a chicken-keeping assistant. Give practical, concise advice. Recommend a vet for serious issues.

Knowledge: {context}

Coop readings: {sensor_context}

Question: {query}

Answer:"""


# Emergency prompt - used when critical sensor alerts are present (used by get_prompt)
BASIC_EMERGENCY_PROMPT = """You are a chicken-keeping assistant. Urgent situation.

Alerts: {sensor_context}

Knowledge: {context}

Question: {query}

Respond with:
1. What's happening (1 sentence)
2. Immediate actions (2-3 steps)
3. When to call a vet

Response:"""


# Prompt for when NO sensor context is needed
SIMPLE_PROMPT = """You are a chicken-keeping assistant. Answer concisely.

Knowledge: {context}

Question: {query}

Answer:"""


def get_prompt(query: str, context: str, sensor_context: str = None, has_critical: bool = False) -> str:
    """
    Select and format the appropriate prompt.
    
    Args:
        query: User's question
        context: Retrieved knowledge base chunks
        sensor_context: Formatted sensor data (if relevant)
        has_critical: Whether there are critical sensor alerts
    
    Returns:
        Formatted prompt string ready for LLM
    """
    
    # If critical alerts, use emergency prompt
    if has_critical and sensor_context:
        return BASIC_EMERGENCY_PROMPT.format(
            sensor_context=sensor_context,
            context=context,
            query=query
        )
    
    # If sensor context is relevant, use main prompt
    if sensor_context:
        return SYSTEM_PROMPT.format(
            sensor_context=sensor_context,
            context=context,
            query=query
        )
    
    # Otherwise, simple prompt without sensors
    return SIMPLE_PROMPT.format(
        context=context,
        query=query
    )


