from pathlib import Path
from config import OUTPUT_DIR


def _safe_path(filename: str) -> Path:
    """Resolve *filename* inside OUTPUT_DIR and reject traversal attempts."""
    if not filename:
        raise ValueError("No filename provided.")

    candidate = (OUTPUT_DIR / filename).resolve()
    if not str(candidate).startswith(str(OUTPUT_DIR.resolve())):
        raise ValueError(f"Path escapes the output directory: {filename}")

    return candidate


def create_file(filename: str, content: str = "") -> str:
    """Create a file inside output/ with optional content.

    Returns a human-readable success / error message.
    """
    try:
        path = _safe_path(filename)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        rel = path.relative_to(OUTPUT_DIR)
        return f"Created file: output/{rel}"
    except Exception as exc:
        return f"ERROR: Could not create file — {exc}"


def create_folder(folder_name: str) -> str:
    """Create a folder inside output/.

    Returns a human-readable success / error message.
    """
    try:
        path = _safe_path(folder_name)
        path.mkdir(parents=True, exist_ok=True)
        rel = path.relative_to(OUTPUT_DIR)
        return f"Created folder: output/{rel}"
    except Exception as exc:
        return f"ERROR: Could not create folder — {exc}"
