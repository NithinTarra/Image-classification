import os
from groq import Groq

api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    raise RuntimeError("GROQ_API_KEY is not configured.")

client = Groq(api_key=api_key)

MODEL_NAME = "openai/gpt-oss-20b"

try:
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are an AI research assistant. "
                    "Provide a short, direct final answer. "
                    "Do not spend excessive tokens reasoning. "
                    "Use no more than 100 words."
                ),
            },
            {
                "role": "user",
                "content": (
                    "A vision model predicts Golden Retriever with a score "
                    "of 0.6074. Ten retrieved reference images are labeled dog. "
                    "Explain what this evidence supports and what it does not prove."
                ),
            },
        ],
        temperature=0.2,
        max_completion_tokens=1000,
    )

    print("\n========== LLM RESPONSE ==========")

    choice = response.choices[0]

    print(choice.message.content or "[No final text generated]")
    print("\nFinish reason:", choice.finish_reason)
    print("Reasoning tokens:",
          response.usage.completion_tokens_details.reasoning_tokens)

except Exception as e:
    print("LLM request failed:", type(e).__name__, str(e))