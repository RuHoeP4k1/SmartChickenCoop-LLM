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

ACTIONABILITY_CRITERIA = “””
Note: Match the standard to the question type. A factual question (“how often do chickens lay?”) is fully
actionable when it gives a clear, usable answer. A problem question (“my chicken won't eat”) is only
actionable when it tells the user something concrete to do. Do NOT penalise a short, direct answer for
lacking numbered steps or sections — format is irrelevant, usefulness is what matters.

Score 1 (Not actionable):
- Gives only vague generalities (“it depends”, “could be many things”) with nothing the user can act on.
- Repeats the question back or provides filler text without any real information.
- After reading the answer a beginner still has no idea what to do or think.

Score 2 (Somewhat actionable):
- Contains at least one genuinely useful piece of information or step, but leaves important gaps.
- For a problem question: mentions what to do but skips key details (e.g. “isolate the bird” with no
  indication of what to watch for or for how long).
- For a factual question: gives a partial or overly hedged answer that still requires the user to look
  elsewhere to get a usable answer.
- Usable in parts, but the user has to fill in the blanks themselves.

Score 3 (Genuinely helpful):
- Directly answers what the user asked in plain, beginner-friendly language.
- For a problem question: tells the user at least one concrete thing to do and what outcome to watch for;
  does not have to be a numbered list — a clear sentence or two is enough.
- For a factual question: gives a specific, correct answer the user can immediately rely on.
- The answer fits the scope of the question — a simple question gets a focused answer, a complex
  situation gets appropriately more detail.
- A hobby keeper reading this answer knows what to do or think next without needing to search further.
“””

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

CORRECTNESS_CRITERIA = “””
Note: We score “safe, sensible, and appropriate” advice for hobby chicken keepers — not scientific depth.
Vet referrals are only relevant when the question involves a genuine health or injury concern. Do NOT
penalise an answer about egg-laying frequency or feed ratios for failing to mention a vet.

Score 1 (Incorrect or harmful):
- Contains clearly wrong facts that could mislead a beginner (e.g., wrong temperature ranges, wrong
  lay rates, wrong feeding guidance).
- Recommends something that could harm the birds: human medications, essential oils, harsh chemicals,
  unsafe dosing, or ignoring symptoms that warrant immediate attention.
- Actively contradicts established poultry welfare or husbandry practice in a way that could cause harm.

Score 2 (Mostly correct but with notable weaknesses):
- Core information is broadly right, but contains a meaningful inaccuracy, an important omission, or
  a claim a beginner could easily misapply.
- For health questions: advice is safe but noticeably incomplete — for example, reassuring the user
  without acknowledging any scenario where the situation could be serious.
- Hedges so heavily (“it could be anything”, “hard to say”) that the answer becomes unreliable even
  though nothing stated is outright wrong.

Score 3 (Correct and appropriate):
- The factual content is accurate and consistent with standard hobby poultry keeping practice.
- Advice is safe: does not recommend medications, unsafe substances, or risky home remedies.
- The answer is calibrated to the question — a routine husbandry question gets a direct factual
  answer; a health question acknowledges limits of home care and mentions professional help only
  when that is genuinely warranted by the situation described.
- A hobby keeper acting on this answer will not be misled or put their birds at risk.
“””
# 👆 Replace everything between the triple quotes, keeping the variable name.
