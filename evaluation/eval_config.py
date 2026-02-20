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
FILL IN YOUR SCORING DESCRIPTION HERE.

Describe:
- Score 1: What does a completely non-actionable answer look like?
- Score 3: What does a partially actionable answer look like?
- Score 5: What does a perfectly actionable answer look like?

Focus on: can a chicken keeper with no vet training follow this immediately?
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
FILL IN YOUR SCORING DESCRIPTION HERE.

Describe:
- Score 1: What does a dangerously wrong answer look like?
- Score 3: What does a partially correct but flawed answer look like?
- Score 5: What does a completely accurate, well-nuanced answer look like?

Focus on: factual accuracy for chicken health, temperature, nutrition, and housing.
"""
# 👆 Replace everything between the triple quotes, keeping the variable name.
