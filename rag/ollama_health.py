"""Utilities to check Ollama server and local model availability."""
from __future__ import annotations

import json
from urllib.error import URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen


def _normalize_base_url(base_url: str) -> str:
    base_url = (base_url or "").strip().rstrip("/")
    if not base_url:
        return "http://127.0.0.1:11434"
    parsed = urlparse(base_url)
    if not parsed.scheme:
        return f"http://{base_url}"
    return base_url


def list_local_models(base_url: str, timeout: float = 3.0) -> set[str]:
    """
    Return locally available Ollama model names from /api/tags.
    Raises RuntimeError if Ollama is unreachable or returns invalid data.
    """
    base = _normalize_base_url(base_url)
    url = f"{base}/api/tags"
    req = Request(url, headers={"Accept": "application/json"})
    try:
        with urlopen(req, timeout=timeout) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except URLError as e:
        raise RuntimeError(f"Cannot reach Ollama at {base}. Is `ollama serve` running?") from e
    except Exception as e:
        raise RuntimeError(f"Failed to query Ollama at {base}: {e}") from e

    models = payload.get("models", [])
    names: set[str] = set()
    for model in models:
        name = model.get("name")
        if isinstance(name, str) and name.strip():
            names.add(name.strip())
    return names


def require_models(base_url: str, required_models: list[str]) -> None:
    """Raise RuntimeError with actionable guidance when required models are missing."""
    names = list_local_models(base_url)
    missing = []
    for model in required_models:
        if not model:
            continue
        # Accept exact names and tagged variants like "model:latest".
        if model in names or any(n.startswith(f"{model}:") for n in names):
            continue
        missing.append(model)
    if missing:
        pulls = "\n".join(f"- ollama pull {m}" for m in missing)
        raise RuntimeError(
            "Missing Ollama model(s): "
            + ", ".join(missing)
            + "\nPull them first:\n"
            + pulls
        )
