"""Configuration & API endpoint discovery.

Everything is overridable via environment variables or a `.env` file in the
project root (see .env.example). Point REDACT_API_URL / REDACT_MODEL at any
OpenAI-compatible vision endpoint — e.g. your own self-hosted Qwen server:

    REDACT_API_URL=http://qwen.internal:8000/v1/chat/completions
    REDACT_MODEL=Qwen/Qwen3-VL-235B-A22B-Instruct
    REDACT_API_KEY=...
"""
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_MODEL = os.environ.get("REDACT_MODEL", "qwen/qwen3-vl-235b-a22b-instruct")
DEFAULT_API_URL = "https://openrouter.ai/api/v1/chat/completions"

_CANDIDATE_ENV_FILES = [
    PROJECT_ROOT / ".env",                                   # project root (recommended)
    Path("C:/agent_linux/.env"),
    Path("/mnt/c/agent_linux/.env"),                         # WSL view of the same
    Path("C:/agent_linux/.env.win"),
    PROJECT_ROOT / "engine" / ".env",
    Path(".env"),                                            # current working dir
]

_KEY_NAMES = ("REDACT_API_KEY", "OPENROUTER_KEY", "QWEN_KEY", "OPENAI_API_KEY")


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


def load_project_env() -> None:
    """Load the first existing .env into os.environ (existing vars win)."""
    for cand in _CANDIDATE_ENV_FILES:
        if cand.is_file():
            for k, v in _parse_env_file(cand).items():
                os.environ.setdefault(k, v)
            return


load_project_env()

# read AFTER .env loading so the file overrides defaults too
API_URL = os.environ.get("REDACT_API_URL", DEFAULT_API_URL)
DEFAULT_MODEL = os.environ.get("REDACT_MODEL", DEFAULT_MODEL)


def get_api_key() -> str:
    """Return the API key: env var first, then the known .env files."""
    for name in _KEY_NAMES:
        key = os.environ.get(name, "")
        if key:
            return key
    for cand in _CANDIDATE_ENV_FILES:
        if cand.is_file():
            vals = _parse_env_file(cand)
            for name in _KEY_NAMES:
                key = vals.get(name, "")
                if key:
                    return key
    raise SystemExit(
        "No API key found. Set REDACT_API_KEY (env var) or add it to .env in the project root.\n"
        "See .env.example for all options."
    )
