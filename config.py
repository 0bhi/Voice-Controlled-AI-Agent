import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

ASSEMBLYAI_API_KEY = os.environ.get("ASSEMBLYAI_API_KEY", "")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
GROQ_MODEL = "llama-3.3-70b-versatile"

BASE_DIR = Path(__file__).parent
OUTPUT_DIR = BASE_DIR / "output"
OUTPUT_DIR.mkdir(exist_ok=True)
