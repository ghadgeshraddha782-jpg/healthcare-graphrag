import os
from groq import Groq

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def get_llm_response(prompt: str) -> str:
    try:
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "user", "content": prompt}
            ],
        )

        return response.choices[0].message.content.strip()

    except Exception as e:
        return f"Error: {str(e)}"
if __name__ == "__main__":
    question = "What is fever?"
    answer = get_llm_response(question)

    print("Q:", question)
    print("A:", answer)