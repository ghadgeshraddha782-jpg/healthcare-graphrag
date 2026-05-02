from src.llm.llm_client import get_llm_response
from src.retrieval.graph_query import query_graph
import time


def build_prompt(user_query: str, graph_data: dict) -> str:
    """
    Create structured prompt using graph context
    """

    prompt = f"""
You are a medical assistant. Use the following structured medical data to answer the question.

User Query:
{user_query}

Graph Context:
Symptoms: {graph_data['symptoms']}
Possible Diseases: {graph_data['possible_diseases']}
Recommended Tests: {graph_data['recommended_tests']}
Treatments: {graph_data['treatments']}

Instructions:
- Give a clear and structured answer
- Explain reasoning
- Be concise
- Add a medical disclaimer

Answer:
"""
    return prompt


def run_graph_rag(query: str) -> dict:
    """
    GraphRAG pipeline
    """

    start_time = time.time()

    # Step 1: Get graph data
    graph_data = query_graph(query)

    # Step 2: Build prompt
    prompt = build_prompt(query, graph_data)

    # Step 3: LLM response
    response = get_llm_response(prompt)

    end_time = time.time()

    return {
        "query": query,
        "graph_data": graph_data,
        "response": response,
        "time_taken": round(end_time - start_time, 2)
    }


# Test run
if __name__ == "__main__":
    question = "I have fever and headache. What could it be?"

    result = run_graph_rag(question)

    print("\n=== GRAPH RAG OUTPUT ===")
    print("Query:", result["query"])
    print("\nGraph Data:", result["graph_data"])
    print("\nResponse:", result["response"])
    print("\nTime Taken:", result["time_taken"], "seconds")