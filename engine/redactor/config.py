"""Configuration & API-key discovery.

Key lookup order:
  1. OPENROUTER_KEY environment variable
  2. C:\\agent_linux\\.env        (Windows layout)
  3. /mnt/c/agent_linux/.env     (WSL view of the same folder)
  4. ./.env                      (project-local)
"""
import os
from pathlib import Path

DEFAULT_MODEL = os.environ.get("REDACT_MODEL", "qwen/qwen3-vl-235b-a22b-instruct")
API_URL = "https://openrouter.ai/api/v1/chat/completions"

_CANDIDATE_ENV_FILES = [
    Path("C:/agent_linux/.env"),
    Path("/mnt/c/agent_linux/.env"),
    Path(__file__).resolve().parent.parent / ".env",
]


def _parse_env_file(path: Path) -> dict:
    values = {}
    try:
        for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            values[k.strip()] = v.strip().strip('"').strip("'")
    except OSError:
        pass
    return values


def get_api_key() -> str:
    """Return the OpenRouter API key, searching env then known .env files."""
    key = os.environ.get("OPENROUTER_KEY", "")
    if key:
        return key
    for cand in _CANDIDATE_ENV_FILES:
        if cand.is_file():
            key = _parse_env_file(cand).get("OPENROUTER_KEY", "")
            if key:
                return key
    raise SystemExit(
        "No OPENROUTER_KEY found. Set the env var or add it to one of:\n  "
        + "\n  ".join(str(p) for p in _CANDIDATE_ENV_FILES)
    )
