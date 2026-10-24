import os
from groq import Groq

MODEL_NAME = "openai/gpt-oss-20b"


def generate_explanation(prediction, evidence, analysis):
    """Generate an explanation grounded in pipeline evidence."""

    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY is missing. Set it in PowerShell."
        )

    client = Groq(api_key=api_key)

    context = {
        "prediction": prediction,
        "retrieved_references": evidence,
        "evidence_analysis": analysis,
    }

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a careful AI research assistant. "
                    "Explain image classification results using "
                    "only the supplied evidence. Distinguish "
                    "supporting evidence from proof."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Analyze these image classification results:\n"
                    f"{context}\n\n"
                    "Explain the prediction, supporting evidence, "
                    "agreement, and uncertainty in fewer than 200 words. "
                    "Do not treat model scores as calibrated probabilities. "
                    "Do not claim similarity proves correctness."
                ),
            },
        ],
        temperature=0.2,
        max_completion_tokens=1500,
    )

    explanation = response.choices[0].message.content

    if not explanation or not explanation.strip():
        raise RuntimeError(
            "The LLM returned no final text. "
            f"Finish reason: {response.choices[0].finish_reason}"
        )

    return explanation.strip()