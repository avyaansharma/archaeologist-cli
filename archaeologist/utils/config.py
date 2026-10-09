import json
import os
from pathlib import Path
from typing import Any, Dict

CONFIG_FILE_NAME = "config.json"


def get_global_config_dir() -> Path:
    """Returns the global user config directory (~/.archaeologist)."""
    home = Path.home()
    config_dir = home / ".archaeologist"
    config_dir.mkdir(parents=True, exist_ok=True)
    # Restrict permissions on POSIX systems
    if os.name != "nt":
        try:
            os.chmod(config_dir, 0o700)
        except OSError:
            pass
    return config_dir


def load_user_config() -> Dict[str, Any]:
    """Loads user configuration from ~/.archaeologist/config.json."""
    config_file = get_global_config_dir() / CONFIG_FILE_NAME
    if config_file.exists():
        try:
            with open(config_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, dict):
                    return data
        except (OSError, json.JSONDecodeError):
            return {}
    return {}


def save_user_config(config: Dict[str, Any]) -> None:
    """Saves user configuration to ~/.archaeologist/config.json with restrictive permissions."""
    config_dir = get_global_config_dir()
    config_file = config_dir / CONFIG_FILE_NAME
    with open(config_file, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    # Restrict config file to owner read/write on POSIX systems
    if os.name != "nt":
        try:
            os.chmod(config_file, 0o600)
        except OSError:
            pass


def sync_env_from_config() -> None:
    """Injects saved config keys into os.environ if not already present."""
    cfg = load_user_config()
    for key in ["GEMINI_API_KEY", "GOOGLE_API_KEY", "GITHUB_TOKEN", "VOYAGE_API_KEY"]:
        if key in cfg and not os.getenv(key):
            os.environ[key] = str(cfg[key])
