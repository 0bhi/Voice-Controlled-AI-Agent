import json
from groq import Groq
from config import GROQ_API_KEY, GROQ_MODEL

client = Groq(api_key=GROQ_API_KEY)

SYSTEM_PROMPT = """You are an intent classifier for a voice-controlled AI agent.

Given the user's spoken text (and optionally prior conversation history), classify
their request into one or more intents. Return valid JSON with the following schema:

{
  "intents": [
    {
      "intent": "<one of: create_file | write_code | summarize | general_chat>",
      "filename": "<suggested filename including extension, or null>",
      "language": "<programming language if applicable, or null>",
      "description": "<concise description of what the user wants>"
    }
  ]
}

Rules:
- "create_file" means the user wants to create an empty file or folder.
- "write_code" means the user wants code generated and saved to a file.
  Always suggest a sensible filename if the user didn't specify one.
- "summarize" means the user wants a piece of text summarized.
- "general_chat" is for anything that doesn't fit the above categories.
- A single utterance may contain MULTIPLE intents (compound commands).
  For example "Summarize this text and save it to summary.txt" is both
  "summarize" and "create_file".
- Always return the JSON object and nothing else.
- Always write all descriptions and filenames in English."""


def classify(text: str, chat_history: list[dict] | None = None) -> list[dict]:
    """Classify the user's transcribed text into structured intents.

    Returns a list of intent dicts. Falls back to a single general_chat
    intent on any failure.
    """
    fallback = [{"intent": "general_chat", "filename": None,
                 "language": None, "description": text}]

    if not GROQ_API_KEY:
        return fallback

    messages: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]

    if chat_history:
        for entry in chat_history[-6:]:
            messages.append({"role": entry.get("role", "user"),
                             "content": entry.get("content", "")})

    messages.append({"role": "user", "content": text})

    try:
        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=messages,
            response_format={"type": "json_object"},
            temperature=0.1,
            max_tokens=1024,
        )

        raw = response.choices[0].message.content
        data = json.loads(raw)
        intents = data.get("intents", [])

        if not intents:
            return fallback

        for intent in intents:
            intent.setdefault("intent", "general_chat")
            intent.setdefault("filename", None)
            intent.setdefault("language", None)
            intent.setdefault("description", text)

        return intents

    except Exception:
        return fallback
