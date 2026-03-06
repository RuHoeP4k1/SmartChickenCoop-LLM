"""
Run the full evaluation pipeline in order:
  1. rag_functions.py  — smoke-test the RAG pipeline
  2. evaluate_rag.py   — RAG vs NO-RAG heuristic comparison
  3. evaluate_retrieval.py — retrieval method comparison
  4. evaluate_ragas.py — RAGAS semantic metrics
  5. evaluate_deepeval.py — DeepEval G-Eval metrics
"""

import os
import subprocess
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)

SCRIPTS = [
    ("RAG pipeline smoke-test", os.path.join(_ROOT, "rag_functions.py")),
    ("RAG vs NO-RAG evaluation", os.path.join(_HERE, "evaluate_rag.py")),
    ("Retrieval evaluation",     os.path.join(_HERE, "evaluate_retrieval.py")),
    ("RAGAS evaluation",         os.path.join(_HERE, "evaluate_ragas.py")),
    ("DeepEval evaluation",      os.path.join(_HERE, "evaluate_deepeval.py")),
]

def run(label, script):
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"  python {script}")
    print(f"{'='*60}\n")
    result = subprocess.run([sys.executable, script], cwd=_ROOT)
    if result.returncode != 0:
        print(f"\n✗ {script} failed (exit code {result.returncode}). Stopping.")
        sys.exit(result.returncode)
    print(f"\n✓ {label} complete.")

for label, script in SCRIPTS:
    run(label, script)

print(f"\n{'='*60}")
print("  All evaluations complete.")
print(f"{'='*60}\n")
