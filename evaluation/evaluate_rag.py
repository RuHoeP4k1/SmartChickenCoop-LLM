"""
Evaluation: RAG vs NO-RAG (heuristic scoring)
Compare performance of RAG pipeline against baseline LLM (no retrieval)

This script can be run standalone for a fast, no-LLM-judge evaluation.
The semantic evaluation scripts (evaluate_ragas.py, evaluate_deepeval.py)
import get_norag_answer() and evaluate_answer_quality() from this file.
"""

import os
import sys
import time
from typing import List, Dict

# ---------------------------------------------------------------------------
# Path setup — makes script runnable from any directory
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _ROOT)

from dotenv import load_dotenv
load_dotenv(dotenv_path=os.path.join(_ROOT, ".env"))

from langchain_ollama import OllamaLLM

from rag_functions import (
    load_documents, split_documents, build_vector_store,
    build_bm25_retriever, answer_query
)
from evaluation_data import TEST_CASES


def get_norag_answer(question: str) -> Dict:
    """
    Get answer using LLM only (no RAG - no knowledge base retrieval).

    Args:
        question: User's question

    Returns:
        Dictionary with answer and timing
    """
    llm = OllamaLLM(
        model=os.getenv("OLLAMA_MODEL", "qwen2.5:1.5b-instruct"),
        temperature=0.7,
        num_predict=400
    )

    prompt = f"""You are a helpful chicken-keeping assistant.

Answer this question based on your knowledge:

Question: {question}

Answer:"""

    start_time = time.time()
    answer = llm.invoke(prompt)
    duration = time.time() - start_time

    return {
        "answer": answer,
        "time": duration
    }


def evaluate_answer_quality(answer: str, expected_topics: List[str]) -> Dict:
    """
    Evaluate answer quality based on simple heuristic metrics.

    Args:
        answer: Generated answer
        expected_topics: Topics that should be covered

    Returns:
        Dictionary with quality scores
    """
    answer_lower = answer.lower()

    # 1. Topic coverage
    topics_covered = sum(
        1 for topic in expected_topics
        if topic.lower() in answer_lower
    )
    topic_score = (topics_covered / len(expected_topics)) * 100 if expected_topics else 0

    # 2. Length (ideal 100-300 words)
    word_count = len(answer.split())
    if 100 <= word_count <= 300:
        length_score = 100
    elif word_count < 100:
        length_score = max(0, word_count)  # Penalize too short
    else:
        length_score = max(0, 100 - (word_count - 300) / 10)  # Penalize too long

    # 3. Actionability (has steps or clear actions)
    has_numbers = any(str(i) in answer for i in range(1, 6))
    has_action_words = any(
        word in answer_lower
        for word in ["should", "must", "need to", "make sure", "check", "ensure"]
    )
    actionable_score = 100 if (has_numbers or has_action_words) else 50

    # Overall score (weighted average)
    overall_score = (topic_score * 0.5 + length_score * 0.3 + actionable_score * 0.2)

    return {
        "topic_coverage": topic_score,
        "length_appropriate": length_score,
        "actionable": actionable_score,
        "overall": overall_score,
        "word_count": word_count,
        "topics_found": topics_covered
    }


def run_evaluation():
    """
    Run complete heuristic evaluation comparing RAG vs NO-RAG.
    Uses TEST_CASES from evaluation_data.py as the question set.
    """
    KB_PATH = os.path.join(_ROOT, "test_docs")
    CHROMA_DIR = os.path.join(_ROOT, "chroma_db")
    RESULTS_FILE = os.path.join(_HERE, "results", "heuristic_results.json")

    print("=" * 80)
    print("RAG vs NO-RAG EVALUATION (heuristic)")
    print("=" * 80)
    print()

    # Setup RAG pipeline
    print("Setting up RAG pipeline...")
    docs = load_documents(KB_PATH)
    chunks = split_documents(docs)
    vectordb = build_vector_store(chunks, persist_dir=CHROMA_DIR, folder_path=KB_PATH)
    bm25 = build_bm25_retriever(chunks)
    print("✓ RAG pipeline ready\n")

    results = {"rag": [], "norag": []}

    for i, test in enumerate(TEST_CASES, 1):
        question = test["question"]
        category = test["category"]
        expected_topics = test["expected_topics"]

        print(f"[{i}/{len(TEST_CASES)}] {category.upper()}")
        print(f"Question: {question}")
        print("-" * 80)

        # Get RAG answer
        print("  Getting RAG answer...")
        start = time.time()
        rag_result = answer_query(
            query=question,
            vectordb=vectordb,
            bm25_retriever=bm25,
            use_sensors=False,
            use_hybrid=True
        )
        rag_time = time.time() - start

        # Get NO-RAG answer
        print("  Getting NO-RAG answer...")
        norag_result = get_norag_answer(question)

        # Evaluate both answers
        rag_quality = evaluate_answer_quality(rag_result["answer"], expected_topics)
        norag_quality = evaluate_answer_quality(norag_result["answer"], expected_topics)

        results["rag"].append({
            "question": question,
            "category": category,
            "answer": rag_result["answer"],
            "sources": rag_result["sources"],
            "time": rag_time,
            "quality": rag_quality
        })
        results["norag"].append({
            "question": question,
            "category": category,
            "answer": norag_result["answer"],
            "time": norag_result["time"],
            "quality": norag_quality
        })

        print(f"\n  RAG:    quality={rag_quality['overall']:.1f}/100  "
              f"topics={rag_quality['topics_found']}/{len(expected_topics)}  "
              f"time={rag_time:.2f}s")
        print(f"  NO-RAG: quality={norag_quality['overall']:.1f}/100  "
              f"topics={norag_quality['topics_found']}/{len(expected_topics)}  "
              f"time={norag_result['time']:.2f}s")
        winner = "RAG" if rag_quality["overall"] > norag_quality["overall"] else "NO-RAG"
        print(f"  Winner: {winner}\n")

    # Summary
    n = len(TEST_CASES)
    rag_avg = sum(r["quality"]["overall"] for r in results["rag"]) / n
    norag_avg = sum(r["quality"]["overall"] for r in results["norag"]) / n
    rag_wins = sum(
        1 for i in range(n)
        if results["rag"][i]["quality"]["overall"] > results["norag"][i]["quality"]["overall"]
    )

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"RAG wins: {rag_wins}/{n}")
    print(f"Avg quality — RAG: {rag_avg:.1f}  NO-RAG: {norag_avg:.1f}  "
          f"diff: {rag_avg - norag_avg:+.1f}")

    import json
    os.makedirs(os.path.dirname(RESULTS_FILE), exist_ok=True)
    with open(RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n✓ Results saved to: {RESULTS_FILE}")


if __name__ == "__main__":
    run_evaluation()
