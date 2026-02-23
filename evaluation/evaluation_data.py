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
        "question": "What ambient temperature are dangerous for chickens?",
        "category": "environment",
        "expected_topics": ["temperature", "heat stress", "temperature-humidity index (THI)", "thermoneutral zone", "danger"],
        "ground_truth": "Chickens perform best in their thermoneutral zone of 19–22 °C. Temperatures above this range will trigger heat stress. "
                        "Lower performances are noticeable between 30-32°C, higher temperatures are thus considered dangerous. "
                        "The temperature-humidity index (THI) classifies conditions as danger at 76–81 and emergency above 81, meaning the birds are at serious thermal risk. "
                        "Practically, when the temperature inside the coop approaches 30°C (90°F): provide shade, cool water and ventilation to cool the chickens.",

    },
    {
        "question": "How many eggs do chickens lay per day?",
        "category": "general",
        "expected_topics": ["egg", "laying", "heat stress", "sexual maturity","day"],
        "ground_truth": "Most laying hens will produce about one egg per day once they reach sexual maturity, especially if they are breeds selected for egg production. "
                        "However, egg laying naturally declines with age and can be reduced by heat stress, which negatively affects egg production. "
                        "For keepers, this means you can expect near-daily eggs from healthy hens in good conditions, but ensure proper housing, cooling, and nutrition to maintain consistent production.",

    },
    {
        "question": "My chicken is limping, what should I do?",
        "category": "health",
        "expected_topics": ["arthritis", "vet", "leg", "isolate"],
        "ground_truth": "If your chicken is limping, first isolate her and check the legs and feet for swelling, warmth, redness, deformities, or a black scab on the foot pad (possible bumblefoot). "
                        "Swollen, hot joints may indicate arthritis, veterinary advice is recommended. "
                        "Provide easy access to feed and water and low perches to reduce strain while you assess the injury. "
                        "Keep bedding clean and dry, as damp or dirty litter can worsen joint infections and leg problems.",
    },
    {
        "question": "What causes chickens to peck each other?",
        "category": "behavior",
        "expected_topics": ["pecking", "light", "stress", "overcrowding"],
        "ground_truth": "Chickens often start pecking due to stress factors such as overcrowding, high stocking density, or uneven access to resources, which increase social conflict and group stress. "
                        "High light intensity or constant light can also worsen injurious pecking behavior. "
                        "Once a bird has a wound or prolapse, the red, inflamed tissue attracts more pecking and can create a vicious cycle. "
                        "To prevent this, provide enough space, reduce light intensity, manage flock weight properly, and cull birds that persistently cannibalize others.",
    },
    {
        "question": "How much space do chickens need?",
        "category": "housing",
        "expected_topics": ["stocking density", "overcrowding", "stress", "outdoor"],
        "ground_truth": "Chickens need enough space to prevent high stocking density, which increases group stress, competition, and injurious pecking. "
                        "A maximum stocking density of 4 birds per m2 is recommended for adult laying hens and layer breeders. "
                        "Overcrowding and insufficient resource allocation (feeders, drinkers, resting areas) raise the risk of feather damage and social conflict. "
                        "Access to a covered veranda or outdoor range can reduce stress and improve welfare. "
                        "Practically, ensure birds are not piling up, huddling excessively, or showing panic reactions, and adjust space, feeder placement, and lighting if you observe these warning signs.",

    },
    {
        "question": "What should I feed laying hens?",
        "category": "nutrition",
        "expected_topics": ["layer ration", "production", "calcium", "shell"],
        "ground_truth": "Laying hens should receive a balanced layer ration starting at 19 weeks of age and continuing through production. "
                        "Because egg production places high calcium demands on the body, provide free-choice oyster shell and ensure adequate vitamin D3 to prevent fragile bones and thin-shelled eggs. "
                        "Feed should be kept dry and protected from contamination, and birds must have constant access to clean, fresh water. "
                        "Practically, watch for thin-shelled eggs, lameness, or a drop in production, these can signal nutritional imbalance and require immediate diet adjustment.",

    },
    {
        "question": "How do I prepare my coop for winter?",
        "category": "seasonal",
        "expected_topics": ["winter", "cold", "insulation", "draft"],
        "ground_truth": "To prepare your coop for winter, ensure it is tightly sealed and well insulated so ventilation can function properly without cold drafts. "
                        "A tight house allows better control of fresh air distribution, reduces temperature fluctuations, and keeps birds comfortable during cold periods. "
                        "Good environmental control also helps remove excess moisture, improving litter quality and lowering ammonia levels. "
                        "Practically, check for air leaks, monitor static pressure and ventilation, keep litter dry, and prevent damp conditions that can increase disease risk.",
    },
    {
        "question": "Why did my chicken stop eating?",
        "category": "health",
        "expected_topics": ["appetite", "illness", "vet", "isolate"],
        "ground_truth": "If your chicken stopped eating, first check for general sick bird signs such as lethargy, ruffled feathers, diarrhea, pale comb, coughing, or discharge from the eyes or nostrils. "
                        "Off-feed behavior can also occur with diseases that cause depression and poor growth, so isolate the bird and monitor closely. "
                        "Moldy or wet feed can lead to health problems and reduced intake, so always keep feed dry and fresh. "
                        "Practically, quarantine the bird, check crop function and droppings, ensure clean water is available, and contact a veterinarian if appetite does not return quickly.",
    },
    {
        "question": "Is 85% humidity bad for chickens?",
        "category": "environment",
        "expected_topics": ["humidity", "ventilation", "health", "heat stress", "moisture"],
        "ground_truth": "Yes, 85% humidity can be harmful, especially when combined with high temperatures, because it increases the risk of heat stress and reduces the birds’ ability to cool themselves. "
                        "High moisture levels also worsen litter conditions, increasing ammonia production and poorer air quality. "
                        "Clumps and caked, wet litter increase the risks of footpad damage, which can lead to footpad dermatitis. "
                        "Practically, improve ventilation, remove wet bedding, and closely monitor birds for panting, open wings, or reduced performance during humid weather.",

    },
    {
        "question": "How do I stop my chickens from escaping?",
        "category": "management",
        "expected_topics": ["fenc", "secure", "confinement", "predator"],  # LET OP de misspelling van fence, dit is express want ik gebruik fencing.
        "ground_truth": "To stop chickens from escaping, keep them confined in secure housing using pasture coops or proper fencing. "
                        "Good biosecurity includes confinement and controlling access points to prevent birds from leaving and outside animals from entering. "
                        "Predators often dig into pens or grab birds through weak spots, so check for gaps, digging areas, and damaged wire regularly. "
                        "Practically, repair holes immediately, reinforce fencing, and ensure doors are closed daily to maintain a safe, controlled environment.",
    },
]
