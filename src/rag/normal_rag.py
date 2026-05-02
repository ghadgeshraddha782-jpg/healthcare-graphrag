from src.llm.llm_client import get_llm_response
import time


def run_normal_rag(query: str) -> dict:
    """
    Baseline system: Direct LLM response
    Returns response + time taken
    """

    start_time = time.time()

    response = get_llm_response(query)

    end_time = time.time()

    return {
        "query": query,
        "response": response,
        "time_taken": round(end_time - start_time, 2)
    }


# Test run
if __name__ == "__main__":
    question = "I have fever and headache. What could it be?"

    result = run_normal_rag(question)

    print("\n=== NORMAL RAG OUTPUT ===")
    print("Query:", result["query"])
    print("Response:", result["response"])
    print("Time Taken:", result["time_taken"], "seconds")