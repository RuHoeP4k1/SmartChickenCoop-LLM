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

We need to prove that this approach actually gives better answers.
To do that, we need to run an evaluation — we ask the AI 10 questions,
score the answers, and compare RAG vs no-RAG.

But for the scoring to mean anything, a human has to define:
- What keywords should appear in a correct answer?
- What does a correct reference answer look like?
- What does "actionable" mean for our users?
- What does "factually correct" mean for chicken keeping?

That is your job.

---

## The Two Files You Need to Edit

### File 1: `evaluation_data.py`

This file contains 10 test questions. For each question, you fill in two things:

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

This file has two text blocks you need to write. No code, just plain English descriptions.

**1. `ACTIONABILITY_CRITERIA`**

Our AI is supposed to give chicken keepers clear steps to follow — not just explain things.
A farmer whose chicken is sick does not need a lecture on chicken biology.
They need to know: check the foot, isolate the bird, call the vet if X happens.

Write a description of what each score level looks like:
- Score 1 (worst): the answer gives no steps at all, just background info
- Score 3 (okay): the answer gives some advice but it is vague or incomplete
- Score 5 (best): the answer gives specific numbered steps the keeper can follow right now

**2. `CORRECTNESS_CRITERIA`**

Write a description of what factually correct vs wrong looks like for chicken keeping advice:
- Score 1 (worst): the advice is actually dangerous — wrong temperatures, harmful feeding, etc.
- Score 3 (okay): mostly right but has some inaccuracies
- Score 5 (best): completely accurate, with correct numbers and nuance

You know enough about poultry science to write this. Think about what kinds of
wrong answers could actually harm the chickens.

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

**Step 3: Fill in `eval_config.py`**

Write the two scoring criteria. A paragraph or two each is enough.
Replace the `FILL_IN` placeholder text inside the triple-quoted strings.

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
- Zero `FILL_IN` strings remaining in `eval_config.py`
- Every `ground_truth` is 2–5 real sentences (not placeholder text)
- Every `expected_topics` has 2–4 actual keywords (reviewed against test_docs/)
- The two criteria descriptions in `eval_config.py` describe specific score levels

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
| `evaluate_deepeval.py` | Runs custom G-Eval scoring automatically | No |
| `results/` | Where output files are saved automatically | No |
| `TEAMMATE_README.md` | This file | No |

---

## One Last Note on the Comparison Setup

Right now we are comparing our RAG system against our own local Qwen model with no documents.
That is the core comparison: does RAG help Qwen give better answers?

There is also a question of whether we should compare against a stronger AI model
(like GPT-4 or Claude) to show where our system sits relative to frontier models.
That decision is pending — we need to discuss it with the supervisor first because
it costs money and changes the scope of the evaluation.

You do not need to do anything about this. Just be aware that this might come up later.

---

## Important: These Questions Do Not Use Sensor Data

The 10 test questions in `evaluation_data.py` are designed to test the AI's
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
The 10 existing questions are all good examples of this.

---

If you have questions, ask in the group chat. Good luck.
