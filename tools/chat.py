from groq import Groq
from config import GROQ_API_KEY, GROQ_MODEL

client = Groq(api_key=GROQ_API_KEY)

SYSTEM_PROMPT = """You are a helpful, friendly AI assistant. Answer the user's
question clearly and concisely. If you don't know the answer, say so honestly.
Always respond in English."""


def general_chat(text: str, chat_history: list[dict] | None = None) -> str:
    """Generate a conversational response using Groq.

    Returns the assistant's reply, or an error message on failure.
    """
    messages: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]

    if chat_history:
        for entry in chat_history[-6:]:
            messages.append({
                "role": entry.get("role", "user"),
                "content": entry.get("content", ""),
            })

    messages.append({"role": "user", "content": text})

    try:
        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=messages,
            temperature=0.7,
            max_tokens=2048,
        )
        return response.choices[0].message.content.strip()
    except Exception as exc:
        return f"ERROR: Chat failed — {exc}"
