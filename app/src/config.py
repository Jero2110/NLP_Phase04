import os
from pathlib import Path

import yaml
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]

PATH_KEYS = [
    ("knowledge", "documents_dir"),
    ("knowledge", "manifest"),
    ("knowledge", "index_dir"),
    ("profiles", "db_path"),
    ("logging", "log_path"),
    ("logging", "feedback_path"),
]


def load_config(path: str | os.PathLike | None = None) -> dict:
    """Loads config.yaml and .env, and resolves relative paths against the project root."""
    load_dotenv(ROOT / ".env")
    with open(path or ROOT / "config.yaml", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    for section, key in PATH_KEYS:
        cfg[section][key] = str(ROOT / cfg[section][key])
    return cfg


def get_secret(name: str, required: bool = False) -> str | None:
    """Reads secrets from the environment only. Never hard-code tokens."""
    value = os.getenv(name)
    if required and not value:
        raise RuntimeError(
            f"Falta la variable de entorno {name}. Defínela en .env o en los secretos de Colab."
        )
    return value