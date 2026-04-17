import json
from pathlib import Path

from config import OUTPUT_DIR

MEMORY_FILE = OUTPUT_DIR / "session_memory.json"
MAX_HISTORY_ITEMS = 200


def load_history() -> list[dict]:
    """Load persisted chat history from disk."""
    if not MEMORY_FILE.exists():
        return []

    try:
        raw = MEMORY_FILE.read_text(encoding="utf-8")
        data = json.loads(raw)
        if isinstance(data, list):
            return [item for item in data if isinstance(item, dict)]
        return []
    except Exception:
        # Keep app usable even if memory file is corrupted.
        return []


def save_history(history: list[dict]) -> None:
    """Persist chat history to disk with a size cap."""
    safe_history = [item for item in history if isinstance(item, dict)]
    trimmed = safe_history[-MAX_HISTORY_ITEMS:]
    MEMORY_FILE.write_text(json.dumps(trimmed, indent=2), encoding="utf-8")


def clear_history() -> None:
    """Delete persisted chat history."""
    if MEMORY_FILE.exists():
        MEMORY_FILE.unlink()
