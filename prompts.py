"""
prompts.py — ChickenCare AI system prompts (research-informed version)

═══════════════════════════════════════════════════════════════════════
DESIGN PHILOSOPHY
═══════════════════════════════════════════════════════════════════════

Every design decision here maps to a specific Score 3 outcome
from eval_config.py. The comments explain WHY, not just WHAT.

METRIC TARGET: Score 3 on both rubric axes every time.

    ACTIONABILITY Score 3 requires:
        ✓ Numbered steps in a logical order
        ✓ Specific doable details (not "check their diet")
        ✓ Monitoring tip (what to watch, how long)
        ✓ Clear "red flags" → vet escalation

    CORRECTNESS Score 3 requires:
        ✓ No medication / dosage instructions
        ✓ No unsafe substances recommended
        ✓ Welfare-first (hydration, cooling, isolation)
        ✓ Vet referral when appropriate

═══════════════════════════════════════════════════════════════════════
OVERALL PROMPT ARCHITECTURE  (inspired by Nuclia RAG Academy)
═══════════════════════════════════════════════════════════════════════

Source: https://nuclia.com/ai/prompting-for-rag/
    "The prompt acts as the instruction manual for the LLM, guiding how
    it should use the provided context to generate the final answer."

Prompt structure used in every variant:

    [1] Persona + hard safety rules   ← anchors model behavior (top)
    [2] Retrieved knowledge base      ← RAG chunks (early, per note [B])
    [3] Live sensor data              ← injected only when relevant
    [4] Embedded mini-example         ← shows output shape to small model
    [5] Question + format rules       ← placed LAST (max attention, note [B])

This "safety → context → question+format" sandwich is intentional.
The model sees the constraints both before and after the content.

═══════════════════════════════════════════════════════════════════════
FEW-SHOT EXAMPLE STRATEGY
═══════════════════════════════════════════════════════════════════════

We embed a single compressed "good answer" example in the format section.
For small models, showing the output shape is more reliable than
describing it abstractly. The example is ~40 tokens — low cost, high
value for format compliance.

Similar pattern used by:
    https://github.com/rajveersinghcse/PetCare-AI-Chatbot
    (pet-care RAG with Ollama — role-anchored format prompting for animal
    health advice, same domain as our project)

═══════════════════════════════════════════════════════════════════════
TOKEN BUDGET  (Gemma 3 1B-4B via Ollama, num_predict=400)
═══════════════════════════════════════════════════════════════════════

    Safety rules + persona:              ~80 tokens
    Format example:                      ~50 tokens
    Knowledge base chunks (3-5 chunks):  ~800-1200 tokens
    Sensor context (when included):      ~40-60 tokens
    Question:                            ~20-40 tokens
    ─────────────────────────────────────────────────────
    Total input:                         ~790-1130 tokens
    Generation budget (num_predict=400): ~300 words output

    The three prompt variants exist to avoid wasting tokens on sensor
    data when it is not relevant to the question.
"""


# =============================================================================
# SHARED SAFETY RULES
# =============================================================================
# Extracted as a constant so the exact same rules appear in all three prompts.
# Consistent wording reduces ambiguity for the model.
# Covers every "Score 1 Unsafe" trigger from CORRECTNESS_CRITERIA:
#   - medications/dosages → blocked
#   - unsafe substances (bleach, essential oils) → blocked
#   - ignoring severe symptoms → blocked by mandatory vet referral
#   - confident claims without safety warnings → blocked by grounding rule

_SAFETY_RULES = """You are ChickenCare AI — a practical assistant for hobby chicken keepers.

Hard rules (follow in every response):
- NEVER suggest medications, dosages, or human medicines for chickens
- NEVER recommend essential oils, vinegar, bleach, or chemical treatments  
- ALWAYS refer to a vet or experienced keeper for serious illness or injury
- ONLY use information from the knowledge base — do not invent facts
- Use plain, beginner-friendly language"""


# =============================================================================
# MAIN PROMPT  — standard questions with sensor context (not critical)
# =============================================================================
# Use when: should_include_sensors() = True, has_critical = False
#
# Key design choices:
#   → Knowledge base placed BEFORE question (note [B])
#   → XML tags for clear section boundaries (note [C])
#   → Grounding instruction embedded in format rules (note [D])
#   → One-line embedded example shows output shape (few-shot strategy)
#   → "monitoring" step built into the format → always hits actionability score

MAIN_PROMPT = """{safety_rules}

<knowledge>
{context}
</knowledge>

<coop_data>
{sensor_block}
</coop_data>

<question>
{query}
</question>

Using ONLY the knowledge base and coop data above, answer the question. If the knowledge base does not contain enough information, say so — do not guess or add facts from outside the knowledge base.

Answer in this format:

**What is happening:** (1 sentence — specific, grounded in the knowledge base or coop data)

**What to do now:** (2-3 numbered action steps — only if the situation requires action. Skip for simple factual questions.)

**Go to a vet or experienced keeper immediately if:** (1-2 specific red flags — only if the question involves health, injury, or critical conditions. Skip for general questions.)

Example (for a heat stress question):
**What is happening:** The coop is above 30°C which is causing heat stress.
**What to do now:** 1. Put cold fresh water inside the coop right now. 2. Open all vents and doors wide. 3. Check every 30 minutes — if panting continues after 1 hour, act urgently.
**Go to a vet if:** A chicken collapses, cannot stand, or stops responding.
"""


# =============================================================================
# SIMPLE PROMPT  — general knowledge questions, no sensor data
# =============================================================================
# Use when: should_include_sensors() = False
#
# Saves ~50 tokens vs MAIN_PROMPT by removing <coop_data> block entirely.
# The vet escalation line is marked "only if relevant" to avoid forcing
# a vet referral on harmless factual questions ("how often do they lay?").
#
# Format stays close to MAIN_PROMPT so evaluation scoring is consistent
# across both question types.

SIMPLE_PROMPT = """{safety_rules}

<knowledge>
{context}
</knowledge>

<question>
{query}
</question>

Using ONLY the knowledge base above, answer the question. If the knowledge base does not contain enough information, say so — do not guess or add facts from outside the knowledge base.

Answer in this format:

**Short answer:** (1-2 sentences — direct, specific, grounded in the knowledge base)

**Details:** (only if the question asks for advice or steps — provide 2-3 numbered action steps. Skip this section entirely for simple factual questions.)

**Call a vet if:** (only if the question involves health or injury — list 1-2 specific red flags. Skip this section entirely for general questions.)
"""


# =============================================================================
# EMERGENCY PROMPT  — critical sensor readings detected
# =============================================================================
# Use when: has_critical = True AND sensor_context is not empty
#
# Design choices vs MAIN/SIMPLE:
#   → Shorter and more direct — sensor alert is already the key context
#   → No "what is happening" analysis — user needs actions, not explanation
#   → Tone stays calm — panic-inducing language would confuse a hobbyist
#     in a stressful moment. "Respond calmly" is an explicit instruction.
#   → Grounding rule repeated — emergency mode carries highest hallucination
#     risk because the model "wants to help more" and may invent solutions
#   → Knowledge base still included — there may be relevant care instructions
#     in the RAG chunks (e.g., heat stress treatment steps)
#
# Note: sensor block goes BEFORE knowledge base here because the critical
# alert IS the primary context — it defines the emergency. The RAG chunks
# are supporting reference. This is the one exception to the "RAG first" rule.

EMERGENCY_PROMPT = """{safety_rules}

URGENT — critical conditions detected in the coop right now:

<coop_data>
{sensor_block}
</coop_data>

<knowledge>
{context}
</knowledge>

<question>
{query}
</question>

Respond calmly and clearly. Use this structure only:

**Do these right now:**
1. [Most critical action first — specific and doable at home]
2. [Second action]
3. [Third action if needed]

**Stop and call a vet immediately if:**
- [Specific symptom or threshold — e.g. "temperature stays above 35°C for 30 min"]
- [Second red flag if relevant]

Do NOT suggest medications, chemicals, or home remedies.
"""


# =============================================================================
# get_prompt()  — selector used by rag_functions.py
# =============================================================================

def get_prompt(
    query: str,
    context: str,
    sensor_context: str = None,
    has_critical: bool = False,
    history: list = None,   # reserved for future conversational context; not yet used
) -> str:
    """
    Select and format the appropriate prompt for the current situation.

    This function is the single entry point called by rag_functions.py.
    The signature is intentionally identical to the previous version so
    the rest of the codebase needs zero changes.

    Args:
        query:          User's question (raw string from the chat interface)
        context:        Concatenated RAG chunk text from vector/BM25 retriever
        sensor_context: Formatted sensor string from sensor_filter.py, or None
                        if sensors were judged irrelevant for this query
        has_critical:   True if get_critical_alerts() returned any alerts
                        (set in rag_functions.py → answer_query())

    Returns:
        A filled prompt string ready to send to OllamaLLM.invoke()

    Decision tree:

        has_critical AND sensor_context  →  EMERGENCY_PROMPT
            Urgent situation: sensor readings in critical range.
            Short, action-first format. Calm tone.

        sensor_context only (not critical)  →  MAIN_PROMPT
            Sensor data is relevant to the question (environment query,
            resource warning, or warning-level readings).
            Full format with <coop_data> block.

        no sensor_context  →  SIMPLE_PROMPT
            General husbandry question. Sensor data not useful here.
            Shorter prompt — saves tokens for the answer.

    Gemma system role note:
        OllamaLLM sends the entire prompt string as the user turn.
        Since Gemma IT models don't support a separate "system" role,
        all constraints (_SAFETY_RULES) are embedded directly in the
        prompt text. This follows Google's official guidance:
        https://ai.google.dev/gemma/docs/core/prompt-structure
    """

    # Path 1: Critical sensor alert — use emergency format
    if has_critical and sensor_context:
        return EMERGENCY_PROMPT.format(
            safety_rules=_SAFETY_RULES,
            sensor_block=sensor_context,
            context=context,
            query=query
        )

    # Path 2: Sensor context available (warning/relevant, not critical)
    if sensor_context:
        return MAIN_PROMPT.format(
            safety_rules=_SAFETY_RULES,
            sensor_block=sensor_context,
            context=context,
            query=query
        )

    # Path 3: General question — no sensor data needed
    return SIMPLE_PROMPT.format(
        safety_rules=_SAFETY_RULES,
        context=context,
        query=query
    )