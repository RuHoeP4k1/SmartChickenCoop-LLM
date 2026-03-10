# Your Task — RAG Evaluation Content

Hey, so this is your piece of the evaluation work.
The code has been built already. Your job is to fill in the content that only a human can write.
No coding required. You can use ChatGPT or any other tool to help you figure out good answers —
that is completely fine and actually encouraged for the ground truth part.

This should take you a few hours if you do it properly.

---

## Why This Even Matters

We built an AI assistant for chicken keepers. The AI uses a technique called RAG —
basically it looks up relevant information from our own documents before answering,
instead of just guessing from memory.

We need to prove that this approach actually gives better answers — and find the
best configuration for it. We do this in two ways:

1. **RAG vs no-RAG comparison** — does retrieving documents help?
2. **Hyperparameter sweep** — which model size, chunk size, k, etc. gives best results?

Both use the same 30 test questions. For the scoring to mean anything, a human has to define:
- What keywords should appear in a correct answer?
- What does a correct reference answer look like?
- What does "actionable" mean for our users?
- What does "factually correct" mean for chicken keeping?

That is your job.

---

## The Two Files You Need to Edit

### File 1: `evaluation_data.py`

This file contains 30 test questions. For each question, you fill in two things:

**1. `expected_topics`**

A list of 2–4 simple keywords that MUST appear in any correct answer.
These are used by the fast scoring method to check: "did the answer even mention the right topics?"

The keywords are pre-filled with sensible defaults. Review them against the actual files
in `test_docs/` — if a document emphasises different terms, update the list to match.

How to think about it: if you asked an expert the question and they gave a good answer,
what words would definitely be in there?

Keep them short and lowercase. Single words or two-word terms only.

Good example for "Is 85% humidity bad for chickens?":
```
["humidity", "ventilation", "moisture"]
```

Bad example:
```
["the humidity in the coop is dangerously high and should be reduced"]
```
(Too long — it won't match text reliably.)

**2. `ground_truth`**

A correct 2–5 sentence answer to the question, written by you.
This is used by the RAGAS evaluation to check how well our AI's answer
compares to a known-correct reference.

⚠️ **Important: base this on our actual documents, not just what you know.**
Open the `test_docs/` folder (one level up from this folder), read the relevant files,
and write the ground truth based on what those documents actually say. If our document
says temperatures above 30°C are dangerous, write 30°C — not 27°C from some website.

If our documents don't cover a topic well, note that in a comment next to the ground_truth.
That is useful information too.

You can use ChatGPT to help you write clean sentences — just make sure the facts
come from our documents, not from ChatGPT's general knowledge.

---

### File 2: `eval_config.py`

**This file is already filled in.** You do not need to edit it unless you think the criteria are wrong or could be improved.

The file contains two scoring criteria used by an AI judge (Gemini 2.0 Flash via OpenRouter free tier) to score our system's answers on a **1–3 scale**.

**1. `ACTIONABILITY_CRITERIA`** — Does the answer tell the keeper what to do?

Key principle: the score reflects *usefulness*, not *format*.
A short direct answer to a simple question scores a 3. A vague answer that leaves the user unsure what to do scores a 1 or 2.
**Do not penalise a short answer for lacking numbered steps** — format is irrelevant, outcome is what matters.

- Score 1 (not actionable): vague generalities, nothing to act on
- Score 2 (partially actionable): has some useful info but leaves important gaps
- Score 3 (genuinely helpful): the keeper knows what to do or think next without needing to search elsewhere

**2. `CORRECTNESS_CRITERIA`** — Is the chicken-keeping advice factually accurate and safe?

Key principle: vet referrals are **only** relevant when the question involves a genuine health or injury concern.
An answer about egg-laying frequency should **not** be penalised for failing to mention a vet.

- Score 1 (incorrect or harmful): wrong facts, dangerous advice, contradicts poultry welfare practice
- Score 2 (mostly correct): core is right but has a meaningful inaccuracy or important omission
- Score 3 (correct and appropriate): accurate, safe, calibrated to the question type

If you think these criteria are wrong or incomplete, leave a comment in the file explaining your reasoning.

---

## How to Work Through This

**Step 1: Read the knowledge base documents**

Go to the `test_docs/` folder (one level up from here, in the project root).
Read the documents. These are the sources our RAG system retrieves from.
The ground_truth answers must come from here.

**Step 2: Fill in `evaluation_data.py`**

Go through each of the 10 questions one by one.
For each one:
- Review the pre-filled `expected_topics` and adjust if needed
- Open the relevant document, find the right information, write a 2–5 sentence ground_truth

Replace all `"FILL_IN"` values with your actual content.

**Step 3: Review `eval_config.py`**

The two scoring criteria are already written. Read them to understand how the AI judge
will score answers — this affects how you write your `ground_truth` answers.
If you think the criteria should be changed, leave a comment in the file.

---

## You Are Also Allowed to Suggest Improvements

If you think the questions we are testing are not the most useful ones — say so.
If you think the way we are measuring "actionability" should be different — propose it.
If you want to add more test questions (we could go from 10 to 15 or 20) — draft them.

Leave your suggestions as comments in the files, like:
```
# SUGGESTION: I think we should also test "what is bumblefoot?" because it comes
# up a lot in the knowledge base and tests whether RAG retrieves disease-specific info.
```

This kind of input is genuinely useful and might end up improving the evaluation.

---

## What "Done" Looks Like

When you are finished, there should be:
- Zero `FILL_IN` strings remaining in `evaluation_data.py`
- Every `ground_truth` is 2–5 real sentences based on `test_docs/` content
- Every `expected_topics` has 2–4 actual keywords (reviewed against test_docs/)
- `eval_config.py` is either unchanged or has your improvement comments added

---

## What You Do NOT Need to Do

- You do not write or run any code
- You do not touch any of the `evaluate_*.py` scripts
- You do not need to install anything
- You do not need to understand how RAGAS or DeepEval work internally

---

## Files in This Folder — Overview

| File | What it is | Do you touch it? |
|---|---|---|
| `evaluation_data.py` | Test questions — review keywords, fill in reference answers | **YES** |
| `eval_config.py` | Scoring criteria — you write the descriptions | **YES** |
| `evaluate_rag.py` | Heuristic scoring utilities + standalone quick-eval | No |
| `evaluate_ragas.py` | Runs RAGAS semantic evaluation automatically | No |
| `evaluate_retrieval.py` | Compares two retrieval methods automatically | No |
| `evaluate_deepeval.py` | Runs custom G-Eval scoring automatically (RAG vs no-RAG) | No |
| `sweep_config.py` | D-Optimal design: parameter grid + design generation | No |
| `sweep.py` | Hyperparameter sweep runner (runs all ~60 configs) | No |
| `sweep_analysis.py` | Analyses sweep results: ANOVA, main effects, ranked table | No |
| `SWEEP_README.md` | Full documentation of the sweep experiment + run instructions | No |
| `results/` | Where output files are saved automatically | No |
| `TEAMMATE_README.md` | This file | No |

---

## One Last Note on the Comparison Setup

Right now we are comparing our RAG system against our own local smollm2 model with no documents.
That is the core comparison: does RAG help the model give better answers?

We are also running a **hyperparameter sweep** to find the best configuration (see `SWEEP_README.md`).
The sweep tests 4 different LLM sizes (including larger models via OpenRouter free API),
2 embedding models, 3 chunk sizes, 3 k values, 2 retrieval modes, and 2 search algorithms —
~60 configurations in total, scored by the same AI judge used here.

The AI judge for both evaluations is **Gemini 2.0 Flash** (Google, via OpenRouter free tier).
It is a different model family from all the models being tested, which avoids self-preference bias.
No costs are incurred for the judge.

---

## Important: These Questions Do Not Use Sensor Data

The 30 test questions in `evaluation_data.py` are designed to test the AI's
**knowledge** — things like correct temperatures, feeding advice, health signs,
and housing guidelines. They are answered using only the knowledge base documents.

The system is also capable of answering questions about live sensor readings —
things like "my coop is currently at 33°C and 80% humidity, what should I do?"
But that kind of evaluation can only be done properly once the physical hardware
is up and running with real sensors attached. That is a separate evaluation that
happens later, with the actual test setup.

**What this means for your questions:** do not add test questions that only make
sense with real sensor data, like "is the current temperature safe?" or "what does
today's humidity reading mean?" Those belong to the later live-system evaluation,
not here.

Stick to questions that any chicken keeper might ask regardless of what the sensors
are reading right now — general knowledge, health, nutrition, housing, behavior.
The 30 existing questions are all good examples of this.

---

If you have questions, ask in the group chat. Good luck.
