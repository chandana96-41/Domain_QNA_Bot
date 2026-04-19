"""Default paths and model names (override in UI or env)."""
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
UPLOADS_DIR = DATA_DIR / "uploads"
CHROMA_DIR = DATA_DIR / "chroma"

DEFAULT_EMBED_MODEL = os.environ.get("OLLAMA_EMBED_MODEL", "nomic-embed-text")
DEFAULT_CHAT_MODEL = os.environ.get("OLLAMA_CHAT_MODEL", "llama3.2")


def ollama_base_url() -> str:
    """Read each time so Streamlit / env overrides apply before each call."""
    return os.environ.get("OLLAMA_BASE_URL", "http://127.0.0.1:11434")

CHROMA_COLLECTION = "domain_docs"


def ensure_dirs() -> None:
    UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
