"""
llm_client.py
Thin wrapper around Groq (default) or OpenAI for chat completions.
Switch provider via config/settings.py → LLM_PROVIDER.
"""

import os
from typing import Optional

# ── Provider imports (lazy) ──────────────────────────────────────────────────
def _get_groq_client():
    from groq import Groq
    return Groq(api_key=os.environ["GROQ_API_KEY"])

def _get_openai_client():
    from openai import OpenAI
    return OpenAI(api_key=os.environ["OPENAI_API_KEY"])


# ── System prompts ────────────────────────────────────────────────────────────
SYSTEM_PROMPT_GRAPHRAG = """
You are a knowledgeable and empathetic medical assistant.
You will receive a list of symptoms provided by the user and a
set of candidate diseases retrieved from a trusted medical knowledge graph.

Your task:
1. Analyze the symptoms in the context of the retrieved diseases.
2. Rank or shortlist the most likely conditions.
3. Explain WHY each condition is relevant based on the symptoms.
4. Suggest general next steps (e.g., see a specialist, tests, lifestyle tips).
5. Always remind the user that this is NOT a substitute for professional diagnosis.

Be concise, factual, and compassionate.
"""

SYSTEM_PROMPT_NORMAL = """
You are a knowledgeable and empathetic medical assistant.
Based solely on your training knowledge, respond to the user's symptoms.
Provide possible conditions, explanations, and general advice.
Always remind the user that this is NOT a substitute for professional diagnosis.
"""


# ── Main LLM client class ─────────────────────────────────────────────────────
class LLMClient:
    def __init__(self, provider: str = "groq", model: Optional[str] = None):
        self.provider = provider.lower()
        if self.provider == "groq":
            self.client = _get_groq_client()
            self.model = model or "llama3-8b-8192"
        elif self.provider == "openai":
            self.client = _get_openai_client()
            self.model = model or "gpt-3.5-turbo"
        else:
            raise ValueError(f"Unsupported provider: {provider}")

    def generate(
        self,
        user_message: str,
        system_prompt: str = SYSTEM_PROMPT_GRAPHRAG,
        temperature: float = 0.3,
        max_tokens: int = 1024,
    ) -> str:
        """Send a prompt and return the assistant's reply as a string."""
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_message},
        ]

        if self.provider == "groq":
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        elif self.provider == "openai":
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )

        return response.choices[0].message.content.strip()


# ── Quick test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    client = LLMClient(provider="groq")
    reply = client.generate(
        user_message="I have fever, headache, and fatigue. What could be wrong?",
        system_prompt=SYSTEM_PROMPT_NORMAL,
    )
    print(reply)
