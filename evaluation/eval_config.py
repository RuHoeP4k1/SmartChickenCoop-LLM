# evaluation/eval_config.py
"""
G-Eval scoring criteria for DeepEval custom metrics.

IMPORTANT FOR THE PERSON FILLING THIS IN:
These are plain-English descriptions that tell the AI judge how to score answers.
You are describing what "good" and "bad" looks like for each metric.
Think about your actual users: hobby chicken keepers who need clear help.

You do NOT need to write code here. Just fill in the text between the triple quotes.
"""

# ---------------------------------------------------------------------------
# ACTIONABILITY CRITERIA
# ---------------------------------------------------------------------------
# This metric measures: "Does the answer give the user clear steps to follow?"
# This matters because our project requires the AI to give ACTION POINTS,
# not just explain things. A user whose chicken is sick needs to know WHAT TO DO.
#
# 👈 TEAMMATE FILLS THIS IN
# Replace the placeholder text with your own scoring description.
# Describe what a score of 1 (worst), 3 (okay), and 5 (best) looks like.
# Keep it focused on chicken keeping — what does a useful answer look like?
#
# You are also welcome to PROPOSE CHANGES to how we measure this.
# If you think the scoring criteria should be different, write a comment below
# explaining your reasoning. This could lead to a better prompt design.
# ---------------------------------------------------------------------------

ACTIONABILITY_CRITERIA = """

Score 1 (Not actionable):
- Mostly generic explanations (“could be stress”, “check their diet”) with no clear next steps.
- No order of actions, no specifics (what to check, how to do it), and no “when to get help” guidance.
- Leaves a beginner unsure what to do today.

Score 2 (Somewhat actionable):
- Gives a few concrete actions (e.g., isolate the bird, refresh water, clean coop), but steps are incomplete or not prioritized.
- Missing key practical details (how long to observe, what signs to watch for, what “clean” means).
- Some guidance is usable, but the user still has to guess.

Score 3 (Highly actionable):
- Clear, beginner-friendly checklist in a sensible order (do this first, then this).
- Includes simple, doable details (what to look for, basic hygiene steps, how to monitor, how long to wait).
- States clear “red flags” and when to contact a vet/experienced keeper immediately.
- Uses plain language and focuses on actions a hobby keeper can realistically do.
"""

# 👆 Replace everything between the triple quotes, keeping the variable name.


# ---------------------------------------------------------------------------
# CORRECTNESS CRITERIA
# ---------------------------------------------------------------------------
# This metric measures: "Is the chicken-keeping advice factually correct?"
# We need this because Qwen 1.5b can confidently say wrong things.
# A good answer should match established poultry science / welfare guidelines.
#
# 👈 TEAMMATE FILLS THIS IN
# Think about: what kinds of wrong answers has the model given in testing?
# What thresholds or facts do you know well from your biosystems background?
# What would a vet or poultry scientist consider dangerously wrong?
#
# Again — if you have ideas for a better way to measure this, write them
# as a comment. Prompt design suggestions are very welcome.
# ---------------------------------------------------------------------------

CORRECTNESS_CRITERIA = """
Note: We are NOT scoring scientific accuracy in depth. We score “safe, sensible, and appropriate advice” for hobby chicken keepers.

Score 1 (Unsafe / inappropriate):
- Encourages risky actions (e.g., give random human meds, antibiotics, essential oils, harsh chemicals, or dosing instructions).
- Tells the user to ignore severe symptoms or delays urgent help.
- Gives confident claims without safety warnings, or advice that could clearly harm the bird.

Score 2 (Mostly safe but imperfect):
- Advice is generally reasonable and welfare-friendly, but has minor questionable points or missing safety caveats.
- Red flags and escalation (vet/urgent care) are mentioned but not clear or not specific enough.
- Some parts are vague or could be misapplied by a beginner.

Score 3 (Safe, sensible, and well-bounded):
- Prioritizes welfare and safety: hydration, warmth/cooling, clean environment, isolation/quarantine when appropriate.
- Avoids giving medication/dosage instructions; does not suggest unsafe substances.
- Clearly explains what to observe and provides “stop and get help now” red flags.
- Practical guidance fits what a hobby keeper can do at home, and refers to a vet/experienced keeper when needed.
"""
# 👆 Replace everything between the triple quotes, keeping the variable name.
