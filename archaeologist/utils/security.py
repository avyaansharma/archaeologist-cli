import os
import re

SHA_REGEX = re.compile(r"^[0-9a-fA-F]{7,40}(\^\d*|~\d*)?$")


def validate_repo_path(repo_path: str) -> str:
    """Validates that repo_path exists and is a valid directory."""
    if not repo_path:
        raise ValueError("Repository path cannot be empty.")
    abs_path = os.path.abspath(repo_path)
    if not os.path.exists(abs_path):
        raise ValueError(f"Repository path '{abs_path}' does not exist.")
    if not os.path.isdir(abs_path):
        raise ValueError(f"Repository path '{abs_path}' is not a directory.")
    return abs_path

def sanitize_sha(sha: str) -> str:
    """Sanitizes git SHA input using strict hex regex allowlist."""
    if not sha or not SHA_REGEX.match(sha):
        raise ValueError(f"Invalid git SHA format: '{sha}'")
    return sha

def sanitize_file_path(base_repo_path: str, user_path: str) -> str:
    """Prevents path traversal attacks by ensuring user_path stays strictly within base_repo_path."""
    abs_base = os.path.abspath(base_repo_path)
    joined = os.path.abspath(os.path.join(abs_base, user_path))
    if joined != abs_base and not joined.startswith(abs_base + os.sep):
        raise ValueError(f"Path traversal detected: '{user_path}' attempts to leave base directory '{abs_base}'.")
    return os.path.relpath(joined, abs_base).replace("\\", "/")

def escape_like(text: str) -> str:
    """Escapes special SQL LIKE wildcard characters % and _."""
    if not text:
        return ""
    return text.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")

