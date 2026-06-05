"""Shared .env loader for the live ETF system (single source of truth)."""

import os

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load_dotenv(path: str = None) -> None:
    """Load KEY=VALUE lines from a .env file into os.environ (no overwrite)."""
    if path is None:
        path = os.path.join(_REPO_ROOT, ".env")
    if not os.path.exists(path):
        return
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))
