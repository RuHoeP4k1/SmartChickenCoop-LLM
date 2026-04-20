# evaluation/shared/eval_config.py
"""
G-Eval scoring criteria for DeepEval custom metrics.

These are plain-English descriptions that tell the AI judge what "good" and
"bad" look like for each metric, framed for our actual users (hobby chicken
keepers who need clear help).

Scale ownership: the numeric scoring scale (0–10 raw → 0–1 normalized) is
defined by the Rubric objects in each script (sweep.py,
evaluate_prompt_variants.py, evaluate_rag_ablation.py). This file owns the
qualitative tier descriptions only — do not reference numeric scores here.
"""

# ---------------------------------------------------------------------------
# ACTIONABILITY CRITERIA
# ---------------------------------------------------------------------------
# This metric measures: "Does the answer give the user clear steps to follow?"
# This matters because our project requires the AI to give ACTION POINTS,
# not just explain things. A user whose chicken is sick needs to know WHAT TO DO.
#
# Note on scale: the numeric scoring scale is defined by the Rubric objects in
# each script (sweep.py, evaluate_prompt_variants.py, evaluate_rag_ablation.py).
# Here you describe what *good* and *bad* look like qualitatively — do not
# reference numeric scores in this text; let the rubric own the numbers.
# ---------------------------------------------------------------------------

ACTIONABILITY_CRITERIA = """
Note: Match the standard to the question type. A factual question (“how often do chickens lay?”) is fully
actionable when it gives a clear, usable answer. A problem question (“my chicken won't eat”) is only
actionable when it tells the user something concrete to do. Do NOT penalise a short, direct answer for
lacking numbered steps or sections — format is irrelevant, usefulness is what matters.

Not actionable:
- Gives only vague generalities (“it depends”, “could be many things”) with nothing the user can act on.
- Repeats the question back or provides filler text without any real information.
- After reading the answer a beginner still has no idea what to do or think.

Partially actionable:
- Contains at least one genuinely useful piece of information or step, but leaves important gaps.
- For a problem question: mentions what to do but skips key details (e.g. “isolate the bird” with no
  indication of what to watch for or for how long).
- For a factual question: gives a partial or overly hedged answer that still requires the user to look
  elsewhere to get a usable answer.
- Usable in parts, but the user has to fill in the blanks themselves.

Genuinely helpful:
- Directly answers what the user asked in plain, beginner-friendly language.
- For a problem question: tells the user at least one concrete thing to do and what outcome to watch for;
  does not have to be a numbered list — a clear sentence or two is enough.
- For a factual question: gives a specific, correct answer the user can immediately rely on.
- The answer fits the scope of the question — a simple question gets a focused answer, a complex
  situation gets appropriately more detail.
- A hobby keeper reading this answer knows what to do or think next without needing to search further.
"""

# ---------------------------------------------------------------------------
# CORRECTNESS CRITERIA
# ---------------------------------------------------------------------------
# This metric measures: "Is the chicken-keeping advice factually correct?"
# A good answer should match established poultry science / welfare guidelines.
#
# Note on scale: numeric scoring is owned by the Rubric objects in each script.
# This text describes only the qualitative tiers — do not reference numeric
# scores here.
# ---------------------------------------------------------------------------

CORRECTNESS_CRITERIA = """
Note: We score “safe, sensible, and appropriate” advice for hobby chicken keepers — not scientific depth.
Vet referrals are only relevant when the question involves a genuine health or injury concern. Do NOT
penalise an answer about egg-laying frequency or feed ratios for failing to mention a vet.

Incorrect or harmful:
- Contains clearly wrong facts that could mislead a beginner (e.g., wrong temperature ranges, wrong
  lay rates, wrong feeding guidance).
- Recommends something that could harm the birds: human medications, essential oils, harsh chemicals,
  unsafe dosing, or ignoring symptoms that warrant immediate attention.
- Actively contradicts established poultry welfare or husbandry practice in a way that could cause harm.

Partially correct:
- Core information is broadly right, but contains a meaningful inaccuracy, an important omission, or
  a claim a beginner could easily misapply.
- For health questions: advice is safe but noticeably incomplete — for example, reassuring the user
  without acknowledging any scenario where the situation could be serious.
- Hedges so heavily (“it could be anything”, “hard to say”) that the answer becomes unreliable even
  though nothing stated is outright wrong.

Correct and appropriate:
- The factual content is accurate and consistent with standard hobby poultry keeping practice.
- Advice is safe: does not recommend medications, unsafe substances, or risky home remedies.
- The answer is calibrated to the question — a routine husbandry question gets a direct factual
  answer; a health question acknowledges limits of home care and mentions professional help only
  when that is genuinely warranted by the situation described.
- A hobby keeper acting on this answer will not be misled or put their birds at risk.
"""


# ---------------------------------------------------------------------------
# HALLUCINATION CRITERIA
# ---------------------------------------------------------------------------
# This metric measures: "Does the answer contain fabricated or incorrect
# poultry-specific claims (wrong numbers, invented practices, statements that
# contradict established poultry welfare practice)?"
#
# Used in the RAG vs no-RAG ablation (evaluate_rag_ablation.py). The reference
# answer (expected_output) anchors "what is correct"; the judge should flag
# specific claims in the actual answer that contradict it or that would be
# harmful if acted on. Generalised or hedged statements that are not claims
# (e.g. "monitor the flock closely") are NOT fabrications even if vague.
#
# Higher score = fewer fabrications (we invert at report time if needed so
# direction matches the other metrics — higher is better).
# ---------------------------------------------------------------------------

HALLUCINATION_CRITERIA = """
Judge ONLY the substantive poultry content of the answer. Hedging, tone, and
structure are out of scope. A "fabricated claim" is a specific factual claim
(numeric threshold, named practice, causal statement) that contradicts the
reference answer OR established poultry welfare practice.

Multiple fabrications:
- Two or more specific claims are wrong or contradict the reference answer.
- At least one incorrect claim is potentially harmful if acted on by a hobby
  keeper (e.g. unsafe temperature range, dangerous feed guidance, invented
  dosage or treatment).
- The answer confidently asserts things that are not supported anywhere.

One notable fabrication:
- Contains one specific claim that is wrong or clearly unsupported.
- OR: states a number, practice, or rule with confidence when the reference
  answer indicates otherwise.
- Other content is broadly correct.

No fabrications:
- All specific claims are either correct (consistent with the reference
  answer or standard practice) or appropriately hedged as uncertain.
- If the answer is brief or partial, absence of content is fine — the metric
  only penalises WRONG content, not missing content.
- "I don't know" or "this depends on your specific flock" is acceptable and
  not a fabrication.
"""
