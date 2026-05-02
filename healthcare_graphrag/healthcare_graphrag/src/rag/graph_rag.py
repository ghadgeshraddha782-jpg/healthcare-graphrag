"""
graph_rag.py
Main GraphRAG pipeline:
  1. Retrieve relevant diseases + matched symptoms from TigerGraph.
  2. Inject the graph context into the LLM prompt.
  3. Return a grounded, explainable medical response.
"""

from src.retrieval.graph_query import GraphQuery
from src.llm.llm_client import LLMClient, SYSTEM_PROMPT_GRAPHRAG
from typing import List, Dict, Any


GRAPHRAG_USER_TEMPLATE = """
User Symptoms: {symptoms}

--- Knowledge Graph Context ---
{graph_context}
--- End of Context ---

Based on the symptoms and the diseases retrieved from the medical knowledge graph above,
please provide:
1. The most likely conditions and why.
2. Key differentiating factors between the top candidates.
3. Recommended next steps (tests, specialists, lifestyle advice).
4. A clear disclaimer that this is not a professional medical diagnosis.
"""


class GraphRAG:
    def __init__(self, provider: str = "groq", top_k: int = 5):
        self.llm       = LLMClient(provider=provider)
        self.graph     = GraphQuery()
        self.top_k     = top_k

    def query(self, symptoms: List[str]) -> Dict[str, Any]:
        """
        Full GraphRAG pipeline.

        Returns:
            {
                "symptoms"      : [...],
                "graph_context" : "...",
                "llm_response"  : "...",
            }
        """
        # ── Step 1: Retrieve from graph ──────────────────────────────────────
        graph_context = self.graph.build_graph_context(symptoms, self.top_k)

        # ── Step 2: Build enriched prompt ────────────────────────────────────
        symptom_str  = ", ".join(symptoms)
        user_message = GRAPHRAG_USER_TEMPLATE.format(
            symptoms=symptom_str,
            graph_context=graph_context,
        )

        # ── Step 3: LLM generates grounded response ──────────────────────────
        llm_response = self.llm.generate(
            user_message,
            system_prompt=SYSTEM_PROMPT_GRAPHRAG,
        )

        return {
            "symptoms":      symptoms,
            "graph_context": graph_context,
            "llm_response":  llm_response,
        }


# ── Quick test ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    pipeline = GraphRAG()
    result   = pipeline.query(["fever", "headache", "fatigue", "rash"])

    print("=== Graph Context ===")
    print(result["graph_context"])
    print("\n=== GraphRAG LLM Response ===")
    print(result["llm_response"])
