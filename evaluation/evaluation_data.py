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
        "expected_topics": ["temperature", "heat stress", "temperature-humidity index (THI)", "thermoneutral zone", "danger"],  # review against test_docs/
        "ground_truth": "Chickens perform best in their thermoneutral zone of 19–22 °C. Temperatures above this range will trigger heat stress"
                        "Lower performances are noticeable between 30-32°C, higher temperatures are thus considered dangerous "
                        "The temperature-humidity index (THI) classifies conditions as danger at 76–81 and emergency above 81, meaning the birds are at serious thermal risk"
                        "Practically, when the temperature inside the coop approaches 30°C (90°F): provide shade, cool water and ventilation to cool the chickens. ",  # 👈 TEAMMATE FILLS THIS — read test_docs/ first
        #   Write 2-5 sentences: what temperature is dangerous? what are the signs?
        #   what should the keeper do? Use numbers from your actual documents.
    },
    {
        "question": "How often do chickens lay eggs?",
        "category": "general",
        "expected_topics": ["egg", "laying", "heat stress", "sexual maturity"],  # review against test_docs/
        "ground_truth": "Most laying hens will produce about one egg per day once they reach sexual maturity, especially if they are breeds selected for egg production."
                        "However, egg laying naturally declines with age and can be reduced by heat stress, which negatively affects egg production."
                        "For keepers, this means you can expect near-daily eggs from healthy hens in good conditions, but ensure proper housing, cooling, and nutrition to maintain consistent production.",  # 👈 TEAMMATE FILLS THIS — read test_docs/ first
        #   Write 2-5 sentences about typical laying frequency and what influences it.
    },
    {
        "question": "My chicken is limping, what should I do?",
        "category": "health",
        "expected_topics": ["arthritis", "vet", "leg", "isolate"],  # review against test_docs/
        "ground_truth": "If your chicken is limping, first isolate her and check the legs and feet for swelling, warmth, redness, deformities, or a black scab on the foot pad (possible bumblefoot)."
                        "Swollen, hot joints may indicate arthritis, veterinary advice is recommended."
                        "Provide easy access to feed and water and low perches to reduce strain while you assess the injury."
                        "Keep bedding clean and dry, as damp or dirty litter can worsen joint infections and leg problems.",  # 👈 TEAMMATE FILLS THIS — read test_docs/ first
    },
    {
        "question": "What causes chickens to peck each other?",
        "category": "behavior",
        "expected_topics": ["pecking", "light", "stress", "overcrowding"],  # review against test_docs/
        "ground_truth": "Chickens often start pecking due to stress factors such as overcrowding, high stocking density, or uneven access to resources, which increase social conflict and group stress."
                        "High light intensity or constant light can also worsen injurious pecking behavior."
                        "Once a bird has a wound or prolapse, the red, inflamed tissue attracts more pecking and can create a vicious cycle."
                        "To prevent this, provide enough space, reduce light intensity, manage flock weight properly, and cull birds that persistently cannibalize others.",  # 👈 TEAMMATE FILLS THIS — read test_docs/ first
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
