from src.rag.normal_rag import run_normal_rag
from src.rag.graph_rag import run_graph_rag


def compare_systems(query: str) -> dict:
    """
    Runs both Normal RAG and GraphRAG and compares results
    """

    print("\n🔹 Running Normal RAG...")
    normal_result = run_normal_rag(query)

    print("🔹 Running GraphRAG...")
    graph_result = run_graph_rag(query)

    # ✅ Accuracy logic (simple heuristic)
    if len(graph_result["response"]) < len(normal_result["response"]):
        accuracy_score = "GraphRAG Better"
    else:
        accuracy_score = "Similar"

    comparison = {
        "query": query,
        "normal_response": normal_result["response"],
        "graph_response": graph_result["response"],
        "normal_time": normal_result["time_taken"],
        "graph_time": graph_result["time_taken"],
        "accuracy": accuracy_score
    }
    return comparison


# Test run
if __name__ == "__main__":
    query = "I have fever and headache. What should I do?"

    result = compare_systems(query)

    print("\n========== COMPARISON ==========")
    print("Query:", result["query"])

    print("\n--- Normal RAG ---")
    print("Time:", result["normal_time"], "sec")
    print("Response:", result["normal_response"])

    print("\n--- GraphRAG ---")
    print("Time:", result["graph_time"], "sec")
    print("Response:", result["graph_response"])
    print("\n--- Comparison Result ---")
    print("Accuracy:", result["accuracy"])
