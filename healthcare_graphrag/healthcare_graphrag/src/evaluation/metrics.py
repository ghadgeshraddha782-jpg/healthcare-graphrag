"""
metrics.py
Evaluation module — compares GraphRAG vs Normal RAG responses.

Metrics implemented:
  - Keyword Coverage  : % of expected disease keywords found in response
  - Response Length   : character / word count
  - Latency           : wall-clock time per query
  - Precision@K       : how many of the top-K retrieved diseases are correct
                        (requires ground-truth labels)
"""

import time
import re
from typing import List, Dict, Any, Optional

from src.rag.normal_rag import NormalRAG
from src.rag.graph_rag  import GraphRAG


# ── Individual metric functions ───────────────────────────────────────────────

def keyword_coverage(response: str, keywords: List[str]) -> float:
    """Fraction of expected keywords present in the response (case-insensitive)."""
    if not keywords:
        return 0.0
    response_lower = response.lower()
    hits = sum(1 for kw in keywords if kw.lower() in response_lower)
    return round(hits / len(keywords), 4)


def response_length(response: str) -> Dict[str, int]:
    return {
        "characters": len(response),
        "words":      len(response.split()),
    }


def precision_at_k(
    retrieved: List[str],
    relevant: List[str],
    k: Optional[int] = None,
) -> float:
    """Standard Precision@K for retrieved disease lists."""
    if k is None:
        k = len(retrieved)
    top_k = retrieved[:k]
    if not top_k:
        return 0.0
    relevant_set = {r.lower() for r in relevant}
    hits = sum(1 for item in top_k if item.lower() in relevant_set)
    return round(hits / k, 4)


# ── Head-to-head comparison ───────────────────────────────────────────────────

def compare(
    symptoms: List[str],
    expected_keywords: List[str] = None,
    relevant_diseases: List[str] = None,
    provider: str = "groq",
) -> Dict[str, Any]:
    """
    Run both Normal RAG and GraphRAG on the same symptom list,
    measure latency and quality metrics, and return a comparison dict.
    """
    normal_rag  = NormalRAG(provider=provider)
    graph_rag   = GraphRAG(provider=provider)
    keywords    = expected_keywords or []
    relevant    = relevant_diseases or []

    # ── Normal RAG ──────────────────────────────────────────────────────────
    t0 = time.time()
    normal_response = normal_rag.query(symptoms)
    normal_latency  = round(time.time() - t0, 3)

    # ── GraphRAG ────────────────────────────────────────────────────────────
    t0 = time.time()
    graph_result   = graph_rag.query(symptoms)
    graph_latency  = round(time.time() - t0, 3)
    graph_response = graph_result["llm_response"]

    # Extract retrieved disease names for Precision@K
    retrieved_diseases = re.findall(
        r"\d+\.\s+(.+?)(?:\n|$)", graph_result["graph_context"]
    )

    report = {
        "symptoms": symptoms,
        "normal_rag": {
            "response":         normal_response,
            "latency_sec":      normal_latency,
            "length":           response_length(normal_response),
            "keyword_coverage": keyword_coverage(normal_response, keywords),
        },
        "graph_rag": {
            "response":         graph_response,
            "graph_context":    graph_result["graph_context"],
            "latency_sec":      graph_latency,
            "length":           response_length(graph_response),
            "keyword_coverage": keyword_coverage(graph_response, keywords),
            "precision_at_k":   precision_at_k(retrieved_diseases, relevant),
        },
    }

    # ── Summary deltas ───────────────────────────────────────────────────────
    report["improvement"] = {
        "keyword_coverage_delta": round(
            report["graph_rag"]["keyword_coverage"]
            - report["normal_rag"]["keyword_coverage"], 4
        ),
        "latency_delta_sec": round(
            graph_latency - normal_latency, 3
        ),
    }

    return report


def print_report(report: Dict[str, Any]) -> None:
    """Pretty-print a comparison report to stdout."""
    sep = "=" * 60
    print(f"\n{sep}")
    print(f"  Evaluation Report — Symptoms: {report['symptoms']}")
    print(sep)

    for label, key in [("Normal RAG", "normal_rag"), ("GraphRAG", "graph_rag")]:
        d = report[key]
        print(f"\n[ {label} ]")
        print(f"  Latency          : {d['latency_sec']}s")
        print(f"  Word count       : {d['length']['words']}")
        print(f"  Keyword coverage : {d['keyword_coverage'] * 100:.1f}%")
        if key == "graph_rag":
            print(f"  Precision@K      : {d['precision_at_k'] * 100:.1f}%")

    imp = report["improvement"]
    print(f"\n[ Delta ]")
    print(f"  Coverage improvement : {imp['keyword_coverage_delta'] * 100:+.1f}%")
    print(f"  Latency overhead     : {imp['latency_delta_sec']:+.3f}s")
    print(sep)


# ── CLI entry point ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    report = compare(
        symptoms=["fever", "headache", "fatigue", "rash"],
        expected_keywords=["malaria", "dengue", "typhoid", "viral infection"],
        relevant_diseases=["Malaria", "Dengue", "Typhoid Fever"],
    )
    print_report(report)
