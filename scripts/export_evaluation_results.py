"""
Export Evaluation Results to CSV
Reads JSON results from evaluation/results/ and produces clean CSVs for paper tables.
"""

import sys, os, csv, json

EVAL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "evaluation", "results")
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "exports")


def ensure_output_dir():
    os.makedirs(OUTPUT_DIR, exist_ok=True)


def write_csv(filename, rows, fieldnames=None):
    if not rows:
        print(f"  {filename}: no data")
        return
    path = os.path.join(OUTPUT_DIR, filename)
    keys = fieldnames or list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"  {filename}: {len(rows)} rows -> {path}")


def load_json(name):
    path = os.path.join(EVAL_DIR, name)
    if not os.path.exists(path):
        print(f"  SKIP {name} (not found)")
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def export_sweep():
    """Latest sweep results -> one row per (config, question)."""
    # Find latest sweep file
    files = sorted(f for f in os.listdir(EVAL_DIR) if f.startswith("sweep_") and f.endswith(".json"))
    if not files:
        print("  No sweep results found")
        return
    data = load_json(files[-1])
    if not data:
        return
    rows = []
    for config in data.get("results", data) if isinstance(data, dict) else data:
        config_name = config.get("config_name", config.get("name", "unknown"))
        for result in config.get("results", []):
            rows.append({
                "config": config_name,
                "question": result.get("question", ""),
                "heuristic_score": result.get("heuristic_score", ""),
                "ragas_score": result.get("ragas_score", ""),
                "answer": result.get("answer", "")[:200],
            })
    write_csv("sweep_results_paper.csv", rows)


def export_prompt_variants():
    """Prompt variant G-Eval results -> one row per (variant, question)."""
    data = load_json("prompt_variant_results.json")
    if not data:
        return
    rows = []
    for item in data if isinstance(data, list) else data.get("results", []):
        rows.append({
            "variant": item.get("variant", ""),
            "question": item.get("question", ""),
            "actionability": item.get("actionability", item.get("actionability_score", "")),
            "correctness": item.get("correctness", item.get("correctness_score", "")),
            "overall": item.get("overall", item.get("overall_score", "")),
            "answer": str(item.get("answer", ""))[:200],
        })
    write_csv("prompt_variants_paper.csv", rows)


def export_sensor_routing():
    """Sensor routing comparison -> one row per (method, scenario, run)."""
    data = load_json("sensor_routing_comparison.json")
    if not data:
        return
    rows = []
    results = data if isinstance(data, list) else data.get("results", [])
    for item in results:
        rows.append({
            "method": item.get("method", ""),
            "scenario": item.get("scenario", item.get("query", "")),
            "expected": item.get("expected", ""),
            "predicted": item.get("predicted", item.get("result", "")),
            "correct": item.get("correct", item.get("match", "")),
            "latency_ms": item.get("latency_ms", item.get("latency", "")),
        })
    write_csv("sensor_routing_paper.csv", rows)


def export_sensor_awareness():
    """Sensor awareness test results -> one row per scenario."""
    data = load_json("sensor_awareness_results.json")
    if not data:
        return
    rows = []
    results = data if isinstance(data, list) else data.get("results", [])
    for item in results:
        rows.append({
            "scenario": item.get("scenario", item.get("name", "")),
            "query": item.get("query", ""),
            "passed": item.get("passed", item.get("pass", "")),
            "details": json.dumps(item.get("checks", item.get("details", "")), default=str)[:200],
        })
    write_csv("sensor_awareness_paper.csv", rows)


if __name__ == "__main__":
    ensure_output_dir()
    print("Exporting evaluation results to CSV...")
    export_sweep()
    export_prompt_variants()
    export_sensor_routing()
    export_sensor_awareness()
    print(f"\nDone. Files in {OUTPUT_DIR}/")
