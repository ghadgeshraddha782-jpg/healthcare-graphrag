def query_graph(user_query: str) -> dict:
    """
    Simulates graph-based retrieval.
    Later we will replace this with TigerGraph queries.
    """

    user_query = user_query.lower()

    # 🔥 Simple rule-based "graph"
    if "fever" in user_query and "headache" in user_query:
        return {
            "symptoms": ["fever", "headache"],
            "possible_diseases": ["viral fever", "dengue", "flu"],
            "recommended_tests": ["blood test", "CBC"],
            "treatments": ["rest", "hydration", "paracetamol"]
        }

    elif "cough" in user_query:
        return {
            "symptoms": ["cough"],
            "possible_diseases": ["common cold", "bronchitis"],
            "recommended_tests": ["chest x-ray"],
            "treatments": ["cough syrup", "steam inhalation"]
        }

    else:
        return {
            "symptoms": [],
            "possible_diseases": ["unknown"],
            "recommended_tests": [],
            "treatments": []
        }


# Test run
if __name__ == "__main__":
    query = "I have fever and headache"

    result = query_graph(query)

    print("\n=== GRAPH QUERY OUTPUT ===")
    for key, value in result.items():
        print(f"{key}: {value}")