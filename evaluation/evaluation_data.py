# evaluation/evaluation_data.py
"""
Shared evaluation data for all evaluation scripts in this folder.

IMPORTANT FOR THE PERSON FILLING THIS IN:
- Read the files in test_docs/ BEFORE writing anything here
- expected_topics: pre-filled with sensible defaults — review against test_docs/
  and adjust to match what your documents actually emphasise
- ground_truth: a correct 2-5 sentence answer based on what test_docs/ actually says
  (not what the internet says — what YOUR documents say)
"""

TEST_CASES = [
    {
        "question": "What temperature is dangerous for chickens?",
        "category": "environment",
        "expected_topics": ["temperature", "heat stress", "cooling"],  # review against test_docs/
        "ground_truth": "FILL_IN",  # 👈 TEAMMATE FILLS THIS — read test_docs/ first
        #   Write 2-5 sentences: what temperature is dangerous? what are the signs?
        #   what should the keeper do? Use numbers from your actual documents.
    },
    {
        "question": "How often do chickens lay eggs?",
        "category": "general",
        "expected_topics": ["eggs", "laying", "frequency"],  # review against test_docs/
        "ground_truth": "FILL_IN",  # 👈 TEAMMATE FILLS THIS — read test_docs/ first
        #   Write 2-5 sentences about typical laying frequency and what influences it.
    },
    {
        "question": "My chicken is limping, what should I do?",
        "category": "health",
        "expected_topics": ["injury", "vet", "leg"],  # review against test_docs/
        "ground_truth": "FILL_IN",  # 👈 TEAMMATE FILLS THIS — read test_docs/ first
    },
    {
        "question": "What causes chickens to peck each other?",
        "category": "behavior",
        "expected_topics": ["pecking", "aggression", "stress"],  # review against test_docs/
        "ground_truth": "FILL_IN",  # 👈 TEAMMATE FILLS THIS — read test_docs/ first
    },
    {
        "question": "How much space do chickens need?",
        "category": "housing",
        "expected_topics": ["space", "coop", "square"],  # review against test_docs/
        "ground_truth": "FILL_IN",  # 👈 TEAMMATE FILLS THIS — read test_docs/ first
        #   Include specific m² numbers if your documents have them.
    },
    {
        "question": "What should I feed laying hens?",
        "category": "nutrition",
        "expected_topics": ["feed", "protein", "calcium"],  # review against test_docs/
        "ground_truth": "FILL_IN",  # 👈 TEAMMATE FILLS THIS — read test_docs/ first
        #   Include protein % and calcium info if your documents mention specific numbers.
    },
    {
        "question": "How do I prepare my coop for winter?",
        "category": "seasonal",
        "expected_topics": ["winter", "cold", "insulation"],  # review against test_docs/
        "ground_truth": "FILL_IN",  # 👈 TEAMMATE FILLS THIS — read test_docs/ first
    },
    {
        "question": "Why did my chicken stop eating?",
        "category": "health",
        "expected_topics": ["appetite", "illness", "vet"],  # review against test_docs/
        "ground_truth": "FILL_IN",  # 👈 TEAMMATE FILLS THIS — read test_docs/ first
    },
    {
        "question": "Is 85% humidity bad for chickens?",
        "category": "environment",
        "expected_topics": ["humidity", "moisture", "health"],  # review against test_docs/
        "ground_truth": "FILL_IN",  # 👈 TEAMMATE FILLS THIS — read test_docs/ first
        #   What does YOUR knowledge base say the safe humidity range is?
    },
    {
        "question": "How do I stop my chickens from escaping?",
        "category": "management",
        "expected_topics": ["fence", "wings", "coop"],  # review against test_docs/
        "ground_truth": "FILL_IN",  # 👈 TEAMMATE FILLS THIS — read test_docs/ first
    },
]
