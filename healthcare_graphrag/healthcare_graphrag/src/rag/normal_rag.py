"""
normal_rag.py
Baseline RAG: sends user symptoms directly to the LLM with
NO graph retrieval. Used for performance comparison.
"""

from src.llm.llm_client import LLMClient, SYSTEM_PROMPT_NORMAL
from typing import List


class NormalRAG:
    def __init__(self, provider: str = "groq"):
        self.llm = LLMClient(provider=provider)

    def query(self, symptoms: List[str]) -> str:
        """
        Build a simple prompt from symptoms and get an LLM response
        without any graph context.
        """
        symptom_str = ", ".join(symptoms)
        user_message = (
            f"I am experiencing the following symptoms: {symptom_str}.\n"
            "What conditions could these symptoms indicate? "
            "Please provide possible diagnoses and recommendations."
        )
        return self.llm.generate(user_message, system_prompt=SYSTEM_PROMPT_NORMAL)


# ── Quick test ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    rag = NormalRAG()
    symptoms = ["fever", "headache", "fatigue", "rash"]
    print("=== Normal RAG Response ===")
    print(rag.query(symptoms))
