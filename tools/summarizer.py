from groq import Groq
from config import GROQ_API_KEY, GROQ_MODEL

client = Groq(api_key=GROQ_API_KEY)

SYSTEM_PROMPT = """You are a concise summarizer. Provide a clear, well-structured
summary of the user's text. Keep it to a few key bullet points or a short paragraph.
Focus on the most important information. Always respond in English."""


def summarize_text(text: str) -> str:
    """Summarize the given text using Groq.

    Returns the summary string, or an error message on failure.
    """
    try:
        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Please summarize the following:\n\n{text}"},
            ],
            temperature=0.3,
            max_tokens=1024,
        )
        return response.choices[0].message.content.strip()
    except Exception as exc:
        return f"ERROR: Summarization failed — {exc}"
