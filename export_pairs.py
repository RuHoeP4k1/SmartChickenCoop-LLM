import json, random, os, sys
sys.path.insert(0, os.path.dirname(__file__))

with open("evaluation/results/prompt_variant_results.json") as f:
    all_results = json.load(f)

from evaluation.evaluate_prompt_variants import _build_pairwise_data
from evaluation.evaluation_data import TEST_CASES

pairs = _build_pairwise_data(all_results, TEST_CASES)

with open("evaluation/results/prompt_pairs.json", "w") as f:
    json.dump(pairs, f, indent=2)

print(len(pairs), "pairs exported")
