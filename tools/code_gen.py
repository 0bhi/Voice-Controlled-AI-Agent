from groq import Groq
from config import GROQ_API_KEY, GROQ_MODEL
from tools.file_ops import create_file

client = Groq(api_key=GROQ_API_KEY)

SYSTEM_PROMPT = """You are an expert programmer. Generate clean, production-quality
code based on the user's description. Return ONLY the code — no markdown fences,
no explanations, no commentary. The code should be complete and runnable."""


def generate_and_save_code(
    description: str,
    filename: str | None = None,
    language: str | None = None,
) -> dict:
    """Generate code via Groq and save it to output/.

    Returns a dict with keys: code, filename, message.
    """
    if not filename:
        ext = _extension_for(language)
        slug = description[:40].strip().replace(" ", "_").lower()
        slug = "".join(c for c in slug if c.isalnum() or c == "_")
        filename = f"{slug}{ext}"

    try:
        prompt = description
        if language:
            prompt = f"Language: {language}\n\n{description}"

        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
            max_tokens=4096,
        )

        code = response.choices[0].message.content.strip()

        if code.startswith("```"):
            lines = code.split("\n")
            lines = lines[1:]  # drop opening fence
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            code = "\n".join(lines)

        save_msg = create_file(filename, code)

        return {
            "code": code,
            "filename": filename,
            "message": save_msg,
        }

    except Exception as exc:
        return {
            "code": "",
            "filename": filename,
            "message": f"ERROR: Code generation failed — {exc}",
        }


def _extension_for(language: str | None) -> str:
    if not language:
        return ".py"
    mapping = {
        "python": ".py", "javascript": ".js", "typescript": ".ts",
        "java": ".java", "c": ".c", "cpp": ".cpp", "c++": ".cpp",
        "go": ".go", "rust": ".rs", "ruby": ".rb", "bash": ".sh",
        "shell": ".sh", "html": ".html", "css": ".css", "sql": ".sql",
        "r": ".r", "php": ".php", "swift": ".swift", "kotlin": ".kt",
    }
    return mapping.get(language.lower(), ".txt")
