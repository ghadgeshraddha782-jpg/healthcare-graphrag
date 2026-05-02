"""
main.py
Entry point for the Healthcare GraphRAG System.
Run: python main.py
"""

from src.utils.helpers import parse_symptoms_from_input
from src.rag.graph_rag import GraphRAG
from src.rag.normal_rag import NormalRAG
from src.config.settings import LLM_PROVIDER, GRAPH_RAG_TOP_K


BANNER = """
╔══════════════════════════════════════════════════════╗
║        Healthcare GraphRAG System  🏥                ║
║  Grounded medical suggestions via Knowledge Graph    ║
╚══════════════════════════════════════════════════════╝
"""


def run_graphrag_mode():
    pipeline = GraphRAG(provider=LLM_PROVIDER, top_k=GRAPH_RAG_TOP_K)
    print("\n[Mode: GraphRAG — graph-grounded responses]\n")

    while True:
        raw = input("Enter your symptoms (comma-separated) or 'quit': ").strip()
        if raw.lower() in ("quit", "exit", "q"):
            break
        symptoms = parse_symptoms_from_input(raw)
        if not symptoms:
            print("Please enter at least one symptom.\n")
            continue

        print("\n⏳ Querying knowledge graph and generating response …\n")
        result = pipeline.query(symptoms)

        print("── Knowledge Graph Context ──────────────────────────────")
        print(result["graph_context"])
        print("\n── Medical Assessment (GraphRAG) ─────────────────────────")
        print(result["llm_response"])
        print("─" * 60 + "\n")


def run_normal_mode():
    rag = NormalRAG(provider=LLM_PROVIDER)
    print("\n[Mode: Normal RAG — LLM only, no graph]\n")

    while True:
        raw = input("Enter your symptoms (comma-separated) or 'quit': ").strip()
        if raw.lower() in ("quit", "exit", "q"):
            break
        symptoms = parse_symptoms_from_input(raw)
        if not symptoms:
            print("Please enter at least one symptom.\n")
            continue

        print("\n⏳ Generating response …\n")
        response = rag.query(symptoms)
        print("── Medical Assessment (Normal RAG) ───────────────────────")
        print(response)
        print("─" * 60 + "\n")


def main():
    print(BANNER)
    print("Select mode:")
    print("  1. GraphRAG  (recommended — graph-grounded)")
    print("  2. Normal RAG (baseline — LLM only)")
    print("  3. Evaluation (compare both side by side)")
    choice = input("\nEnter 1, 2, or 3: ").strip()

    if choice == "1":
        run_graphrag_mode()
    elif choice == "2":
        run_normal_mode()
    elif choice == "3":
        from src.evaluation.metrics import compare, print_report
        raw = input("Enter symptoms for evaluation: ").strip()
        symptoms = parse_symptoms_from_input(raw)
        kw_raw   = input("Expected keywords/diseases (comma-separated, optional): ").strip()
        keywords = [k.strip() for k in kw_raw.split(",") if k.strip()]
        report   = compare(symptoms, expected_keywords=keywords)
        print_report(report)
    else:
        print("Invalid choice. Exiting.")


if __name__ == "__main__":
    main()
