# evaluation/shared/evaluation_data.py
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
    {
        "question": "How often should I clean my chicken coop?",
        "category": "management",
        "expected_topics": ["clean", "coop", "litter"],
        "ground_truth": "Chicken coops should be cleaned regularly to prevent disease organisms from building up in the environment."
                        "Daily tasks include removing droppings, refreshing water, and checking feed."
                        "Litter should be replaced periodically and the coop should receive a thorough deep cleaning at least once per year."
                        "Keeping bedding dry and clean helps reduce disease and maintain flock health.",
    },
    {
        "question": "How much water do chickens need per day?",
        "category": "nutrition",
        "expected_topics": ["water", "hydration", "clean", "daily"],
        "ground_truth": "Chickens must have constant access to fresh, clean drinking water throughout the day."
                        "Waterers should be cleaned and refilled daily to prevent bacteria or algae buildup."
                        "Feeders and waterers should contain enough supply so the flock never runs out."
                        "Reduced water intake can quickly affect egg production and overall flock health.",
    },
    {
        "question": "Why are my chickens losing feathers?",
        "category": "health",
        "expected_topics": ["feathers", "parasites", "pecking", "mites"],
        "ground_truth": "Feather loss can occur due to parasites, disease, or aggressive pecking within the flock."
                        "External parasites such as lice or mites can damage feathers and reduce production."
                        "Feather pecking may also remove feathers when birds are stressed or overcrowded."
                        "Check birds regularly for parasites and manage space and flock conditions to prevent feather damage.",
    },
    {
        "question": "How can I tell if my chicken is sick?",
        "category": "health",
        "expected_topics": ["symptoms", "lethargy", "vet", "egg"],
        "ground_truth": "Chickens often hide illness, so watch for warning signs such as lethargy, ruffled feathers, diarrhea, or reduced egg production."
                        "Other symptoms include coughing, sneezing, discharge from the eyes or nostrils, and weight loss."
                        "Daily observation helps detect these problems early."
                        "If multiple birds show symptoms, seek veterinary advice to diagnose the illness.",
    },
    {
        "question": "What predators attack backyard chickens?",
        "category": "management",
        "expected_topics": ["predator", "raccoon", "hawk", "fox"],
        "ground_truth": "Many predators prey on backyard chickens including raccoons, foxes, hawks, owls, dogs, and rats."
                        "Signs of predator attacks include scattered feathers, missing birds, or partially eaten carcasses."
                        "Some predators hunt during the day while others attack at night."
                        "Recognizing predator signs helps chicken keepers improve coop protection.",
    },
    {
        "question": "How can I protect my chickens from predators?",
        "category": "management",
        "expected_topics": ["predator", "fenc", "coop", "secur"], #misspelled on purpose
        "ground_truth": "Secure housing and fencing are essential to protect chickens from predators."
                        "Birds should be kept in enclosed coops or fenced runs that prevent animals from digging or reaching through wire."
                        "Regularly inspect the coop for holes, weak spots, or damaged fencing."
                        "Fixing these vulnerabilities helps keep predators out and chickens safe.",
    },
    {
        "question": "Why are my chickens not laying eggs?",
        "category": "general",
        "expected_topics": ["egg", "production", "stress", "nutrition"],
        "ground_truth": "Egg production can decrease due to illness, stress, or poor nutrition."
                        "Sick birds may also show signs such as lethargy, weight loss, or reduced appetite."
                        "Nutritional deficiencies or poor housing conditions can also lower egg production."
                        "Maintaining proper diet, housing, and health checks helps support consistent egg laying.",
    },
    {
        "question": "How do I introduce new chickens to my flock?",
        "category": "management",
        "expected_topics": ["quarantine", "isolate", "flock", "disease"],
        "ground_truth": "New chickens should be isolated from the existing flock for several weeks before introduction."
                        "This quarantine period allows you to monitor the birds for signs of disease."
                        "After quarantine, introduce them gradually to reduce stress and aggression."
                        "Proper isolation helps prevent disease transmission within the flock.",
    },
    {
        "question": "What are the signs of heat stress in chickens?",
        "category": "environment",
        "expected_topics": ["heat stress", "panting", "wings", "temperature"],
        "ground_truth": "Chickens suffering from heat stress may pant rapidly and hold their wings away from their bodies."
                        "They may also eat less and show reduced productivity."
                        "If many birds show these signs, the environment is likely too hot."
                        "Providing shade, ventilation, and cool water helps reduce heat stress.",
    },
    {
        "question": "How can I keep my chickens cool in summer?",
        "category": "seasonal",
        "expected_topics": ["summer", "cool", "shade", "ventilation"],
        "ground_truth": "To keep chickens cool in summer, provide shade, fresh drinking water, and good airflow."
                        "Ventilation helps remove heat from the coop and keeps temperatures manageable."
                        "Avoid overcrowding so birds can spread out and stay cool."
                        "These practices help reduce heat stress during hot weather.",
    },
    {
        "question": "How can I prevent diseases in my flock?",
        "category": "health",
        "expected_topics": [ "sanitation", "disease", "prevention"],
        "ground_truth": "Disease prevention relies on good biosecurity practices such as isolating new birds and limiting outside contact."
                        "Clean housing, equipment, and proper sanitation reduce the spread of pathogens."
                        "Controlling rodents and insects also helps prevent disease transmission."
                        "Following these practices helps maintain a healthy flock.",
    },
    {
        "question": "Why are my chickens making loud noises?",
        "category": "behavior",
        "expected_topics": ["noise", "stress", "predator", "disturbance"],
        "ground_truth": "Chickens may make loud noises due to stress or disturbances in their environment."
                        "Predator threats, sudden noises, or flock conflicts can trigger alarm calls."
                        "Birds may also become noisy during stressful conditions such as overcrowding."
                        "Ensuring safe housing and reducing disturbances helps keep birds calm.",
    },
    {
        "question": "What bedding should I use in my chicken coop?",
        "category": "housing",
        "expected_topics": ["bedding", "litter", "moisture", "coop"],
        "ground_truth": "Chicken coop bedding should be a loose and absorbent material such as wood shavings."
                        "Good litter absorbs moisture and provides insulation from the floor."
                        "It also allows chickens to perform natural behaviors like scratching and dust bathing."
                        "Wet bedding should be removed promptly to maintain coop hygiene.",
    },
    {
        "question": "How often should I check my chickens for parasites?",
        "category": "health",
        "expected_topics": ["parasite", "mite", "lice", "inspect"],
        "ground_truth": "Chickens should be checked regularly for parasites, ideally at least once per week."
                        "External parasites such as mites or lice can damage feathers and reduce productivity."
                        "Early detection prevents infestations from spreading through the flock."
                        "Routine health inspections help maintain overall flock health.",
    },
    {
        "question": "Why are my chickens fighting each other?",
        "category": "behavior",
        "expected_topics": ["pecking order", "fight", "space", "stress"],
        "ground_truth": "Chickens establish a social hierarchy known as the pecking order."
                        "Fighting may occur when birds compete for dominance or when resources are limited."
                        "Overcrowding and stress can increase aggressive behavior."
                        "Providing adequate space and resources helps reduce fighting within the flock.",
    },
    {
        "question": "How many chickens should I keep in one flock?",
        "category": "management",
        "expected_topics": ["flock size", "space", "stocking density", "welfare"],
        "ground_truth": "The number of chickens in a flock should match the available space and housing conditions."
                        "High stocking density increases stress and competition between birds."
                        "Adequate space and resources improve welfare and reduce aggressive behavior."
                        "Keepers should adjust flock size to maintain healthy living conditions.",
    },
    {
        "question": "When do chickens start laying eggs?",
        "category": "general",
        "expected_topics": ["sexual maturity", "egg", "laying", "produc"],
        "ground_truth": "Most hens begin laying eggs once they reach sexual maturity, usually around 19 weeks of age."
                        "At this stage they require a balanced layer diet to support egg production."
                        "Proper nutrition and health management help maintain consistent laying."
                        "Healthy hens can produce eggs frequently under good conditions.",
    },
    {
        "question": "How do I keep rodents away from my chicken coop?",
        "category": "management",
        "expected_topics": ["rodent", "feed", "control", "coop"],
        "ground_truth": "Rodents are attracted to chicken coops because they provide food, water, and shelter."
                        "Rats and mice can contaminate feed and cause significant losses."
                        "Remove spilled feed, seal entry holes, and maintain rodent control programs."
                        "Regular inspection helps prevent rodent infestations.",
    },
    {
        "question": "How can I tell if my chicken has parasites?",
        "category": "health",
        "expected_topics": ["parasite", "symptom", "feather", "weight loss"],
        "ground_truth": "Parasites may cause symptoms such as weight loss, droopiness, and feather damage."
                        "External parasites may appear as insects or specks around feathers or the vent area."
                        "Internal parasites can cause diarrhea or weakness."
                        "Regular health checks help detect parasite problems early.",
    },
    {
        "question": "Why are my chickens laying thin-shelled eggs?",
        "category": "nutrition",
        "expected_topics": ["egg", "calcium", "vitamin D", "nutrition", "diet"],
        "ground_truth": "Thin-shelled eggs are often caused by insufficient calcium or vitamin D in the diet."
                        "Egg production requires large amounts of calcium to form strong shells."
                        "If the diet lacks calcium, hens may draw it from their bones."
                        "Providing oyster shell and balanced nutrition helps maintain strong eggshell quality.",
    },
]
